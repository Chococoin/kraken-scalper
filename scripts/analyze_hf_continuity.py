#!/usr/bin/env python3
"""
Analyze data continuity in HuggingFace dataset.

Fetches parquet files from HuggingFace and analyzes gaps in data collection.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from huggingface_hub import HfApi, hf_hub_download
import tempfile
import os

def fetch_hf_files(repo_id: str, pattern: str = "*.parquet") -> list:
    """List all parquet files in HuggingFace repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    return [f for f in files if f.endswith('.parquet')]

def download_and_analyze(repo_id: str, file_path: str, cache_dir: str) -> pd.DataFrame:
    """Download a parquet file and extract timestamps."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        df = pd.read_parquet(local_path)
        if 'ts' in df.columns:
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            return df[['datetime']].copy()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return pd.DataFrame()

def analyze_continuity(repo_id: str, category: str = "crypto", data_type: str = "ticker",
                       max_files: int = 50) -> dict:
    """Analyze data continuity for a specific category and data type."""

    print(f"Fetching file list from {repo_id}...")
    all_files = fetch_hf_files(repo_id)

    # Filter files by category and data type
    pattern = f"{category}/{data_type}/"
    files = sorted([f for f in all_files if pattern in f])[-max_files:]

    print(f"Found {len(files)} files matching {pattern}")

    if not files:
        return {}

    all_timestamps = []

    with tempfile.TemporaryDirectory() as cache_dir:
        for i, file_path in enumerate(files):
            print(f"  [{i+1}/{len(files)}] {file_path}")
            df = download_and_analyze(repo_id, file_path, cache_dir)
            if not df.empty:
                all_timestamps.extend(df['datetime'].tolist())

    if not all_timestamps:
        return {}

    # Sort and analyze
    all_timestamps = sorted(set(all_timestamps))

    # Calculate gaps
    gaps = []
    for i in range(1, len(all_timestamps)):
        delta = (all_timestamps[i] - all_timestamps[i-1]).total_seconds()
        if delta > 60:  # Gap > 1 minute
            gaps.append({
                'start': all_timestamps[i-1],
                'end': all_timestamps[i],
                'duration_minutes': delta / 60
            })

    return {
        'timestamps': all_timestamps,
        'gaps': gaps,
        'total_records': len(all_timestamps),
        'start_time': min(all_timestamps),
        'end_time': max(all_timestamps),
        'total_duration_hours': (max(all_timestamps) - min(all_timestamps)).total_seconds() / 3600
    }

def plot_continuity(analysis: dict, title: str = "Data Continuity", output_path: str = None):
    """Create visualization of data continuity."""

    if not analysis or not analysis.get('timestamps'):
        print("No data to plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{title}\n{analysis['start_time'].strftime('%Y-%m-%d %H:%M')} to {analysis['end_time'].strftime('%Y-%m-%d %H:%M')}",
                 fontsize=14, fontweight='bold')

    timestamps = analysis['timestamps']
    gaps = analysis['gaps']

    # 1. Timeline with gaps highlighted
    ax1 = axes[0]
    ax1.set_title(f"Data Timeline ({analysis['total_records']:,} records, {analysis['total_duration_hours']:.1f} hours)")

    # Plot data presence as green bars
    for i, ts in enumerate(timestamps):
        ax1.axvline(ts, color='green', alpha=0.1, linewidth=0.5)

    # Highlight gaps in red
    for gap in gaps:
        ax1.axvspan(gap['start'], gap['end'], color='red', alpha=0.3)

    ax1.set_xlim(analysis['start_time'], analysis['end_time'])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.set_ylabel('Data Present')
    ax1.set_yticks([])

    # 2. Gap duration histogram
    ax2 = axes[1]
    if gaps:
        gap_durations = [g['duration_minutes'] for g in gaps]
        ax2.hist(gap_durations, bins=30, color='red', alpha=0.7, edgecolor='darkred')
        ax2.set_xlabel('Gap Duration (minutes)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f"Gap Distribution ({len(gaps)} gaps, avg: {sum(gap_durations)/len(gap_durations):.1f} min)")
        ax2.axvline(sum(gap_durations)/len(gap_durations), color='black', linestyle='--', label='Mean')
    else:
        ax2.text(0.5, 0.5, 'No gaps detected!', ha='center', va='center', fontsize=14, color='green')
        ax2.set_title("Gap Distribution")

    # 3. Data density over time (records per hour)
    ax3 = axes[2]
    df_ts = pd.DataFrame({'datetime': timestamps})
    df_ts['hour'] = df_ts['datetime'].dt.floor('H')
    hourly_counts = df_ts.groupby('hour').size()

    ax3.bar(hourly_counts.index, hourly_counts.values, width=0.04, color='blue', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Records per Hour')
    ax3.set_title(f"Data Density (avg: {hourly_counts.mean():.0f} records/hour)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax3.axhline(hourly_counts.mean(), color='red', linestyle='--', alpha=0.7, label='Mean')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()

def print_gap_report(analysis: dict):
    """Print detailed gap report."""
    if not analysis:
        print("No data to analyze")
        return

    print("\n" + "="*60)
    print("DATA CONTINUITY REPORT")
    print("="*60)
    print(f"Total records: {analysis['total_records']:,}")
    print(f"Time range: {analysis['start_time']} to {analysis['end_time']}")
    print(f"Duration: {analysis['total_duration_hours']:.2f} hours")

    gaps = analysis['gaps']
    print(f"\nGaps detected: {len(gaps)}")

    if gaps:
        total_gap_minutes = sum(g['duration_minutes'] for g in gaps)
        coverage = 100 * (1 - total_gap_minutes / (analysis['total_duration_hours'] * 60))

        print(f"Total gap time: {total_gap_minutes:.1f} minutes ({total_gap_minutes/60:.2f} hours)")
        print(f"Data coverage: {coverage:.1f}%")

        print(f"\nTop 10 largest gaps:")
        sorted_gaps = sorted(gaps, key=lambda x: x['duration_minutes'], reverse=True)[:10]
        for i, gap in enumerate(sorted_gaps, 1):
            print(f"  {i}. {gap['start'].strftime('%Y-%m-%d %H:%M')} -> {gap['end'].strftime('%Y-%m-%d %H:%M')} ({gap['duration_minutes']:.1f} min)")
    else:
        print("No significant gaps detected!")
        print("Data coverage: 100%")

def main():
    parser = argparse.ArgumentParser(description='Analyze HuggingFace dataset continuity')
    parser.add_argument('--repo', type=str, default='Abraxasccs/kraken-market-data',
                        help='HuggingFace repo ID')
    parser.add_argument('--category', type=str, default='crypto', choices=['crypto', 'stocks'])
    parser.add_argument('--type', type=str, default='ticker',
                        choices=['ticker', 'ohlc', 'trade', 'book'])
    parser.add_argument('--max-files', type=int, default=50,
                        help='Maximum number of files to analyze')
    parser.add_argument('--output', type=str, default='reports/hf_continuity.png',
                        help='Output path for plot')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    # Analyze
    print(f"Analyzing {args.category}/{args.type} from {args.repo}")
    analysis = analyze_continuity(
        args.repo,
        category=args.category,
        data_type=args.type,
        max_files=args.max_files
    )

    # Report
    print_gap_report(analysis)

    # Plot
    if not args.no_plot and analysis:
        plot_continuity(
            analysis,
            title=f"HuggingFace Data Continuity: {args.category}/{args.type}",
            output_path=args.output
        )

if __name__ == "__main__":
    main()
