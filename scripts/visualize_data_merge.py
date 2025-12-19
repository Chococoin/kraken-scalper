#!/usr/bin/env python3
"""
Visualize data merge potential between Kraken REST and WebSocket data.

Shows coverage, gaps, and data quality for a hypothetical 3-month merge.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from collections import defaultdict

def load_kraken_rest_data(data_dir: Path, pair: str) -> dict:
    """Load all available Kraken REST data for a pair."""
    pair_safe = pair.replace("/", "_")
    kraken_dir = data_dir / "kraken" / pair_safe

    data = {}
    if not kraken_dir.exists():
        return data

    for parquet_file in kraken_dir.glob("*.parquet"):
        interval = int(parquet_file.stem)
        df = pd.read_parquet(parquet_file)
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.sort_values('datetime')
        data[interval] = df

    return data

def load_websocket_data(data_dir: Path, pair: str, category: str = "crypto") -> pd.DataFrame:
    """Load WebSocket ticker data from local parquet files."""
    ticker_dir = data_dir / category / "ticker"

    if not ticker_dir.exists():
        return pd.DataFrame()

    all_dfs = []
    for date_dir in sorted(ticker_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for parquet_file in date_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                if 'pair' in df.columns:
                    df = df[df['pair'] == pair]
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {parquet_file}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['datetime'] = pd.to_datetime(combined['ts'], unit='ms')
    combined = combined.sort_values('datetime').drop_duplicates(subset=['ts'])

    return combined

def analyze_coverage(df: pd.DataFrame, expected_interval_seconds: int = 60) -> dict:
    """Analyze data coverage and gaps."""
    if df.empty:
        return {'coverage': 0, 'gaps': [], 'total_records': 0}

    df = df.sort_values('datetime')
    timestamps = df['datetime'].tolist()

    # Calculate gaps
    gaps = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i-1]).total_seconds()
        if delta > expected_interval_seconds * 1.5:  # Gap if > 1.5x expected
            gaps.append({
                'start': timestamps[i-1],
                'end': timestamps[i],
                'duration_seconds': delta
            })

    # Calculate coverage
    if len(timestamps) > 1:
        total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
        gap_duration = sum(g['duration_seconds'] for g in gaps)
        coverage = 100 * (1 - gap_duration / total_duration) if total_duration > 0 else 0
    else:
        coverage = 0

    return {
        'coverage': coverage,
        'gaps': gaps,
        'total_records': len(timestamps),
        'start': timestamps[0] if timestamps else None,
        'end': timestamps[-1] if timestamps else None,
        'duration_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600 if len(timestamps) > 1 else 0
    }

def calculate_data_quality(df: pd.DataFrame) -> dict:
    """Calculate data quality metrics."""
    if df.empty:
        return {'completeness': 0, 'consistency': 0, 'freshness': 0}

    # Completeness: percentage of non-null values in key columns
    key_cols = ['open', 'high', 'low', 'close', 'volume'] if 'open' in df.columns else ['last', 'volume']
    key_cols = [c for c in key_cols if c in df.columns]

    if key_cols:
        completeness = 100 * (1 - df[key_cols].isnull().sum().sum() / (len(df) * len(key_cols)))
    else:
        completeness = 0

    # Consistency: check for anomalies (price spikes > 10%)
    if 'close' in df.columns:
        returns = df['close'].pct_change().abs()
        anomalies = (returns > 0.1).sum()
        consistency = 100 * (1 - anomalies / len(df))
    elif 'last' in df.columns:
        returns = df['last'].pct_change().abs()
        anomalies = (returns > 0.1).sum()
        consistency = 100 * (1 - anomalies / len(df))
    else:
        consistency = 100

    # Freshness: how recent is the data
    if 'datetime' in df.columns:
        latest = df['datetime'].max()
        age_hours = (datetime.now() - latest.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
        freshness = max(0, 100 - age_hours * 2)  # Lose 2% per hour of staleness
    else:
        freshness = 0

    return {
        'completeness': completeness,
        'consistency': consistency,
        'freshness': freshness,
        'overall': (completeness + consistency + freshness) / 3
    }

def create_merge_visualization(kraken_data: dict, ws_data: pd.DataFrame, pair: str, output_path: str = None):
    """Create visualization of data merge potential."""

    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(f'Data Coverage Analysis: {pair}\nHypothetical 3-Month Merge View',
                 fontsize=14, fontweight='bold')

    # Define time range: 3 months back from now
    now = datetime.now()
    three_months_ago = now - timedelta(days=90)

    colors = {
        'kraken_1m': '#2ecc71',    # Green
        'kraken_1h': '#3498db',    # Blue
        'kraken_1d': '#9b59b6',    # Purple
        'websocket': '#e74c3c',    # Red
        'gap': '#e74c3c',          # Red for gaps
        'overlap': '#f39c12',      # Orange for overlap
    }

    # =========================================================================
    # Panel 1: Timeline Coverage
    # =========================================================================
    ax1 = axes[0]
    ax1.set_title('Data Source Coverage Timeline', fontsize=12, fontweight='bold')

    y_positions = {'Kraken 1m (REST)': 4, 'Kraken 1h (REST)': 3, 'Kraken 1d (REST)': 2, 'WebSocket': 1}

    # Plot Kraken REST data
    for interval, df in kraken_data.items():
        if df.empty:
            continue
        start = df['datetime'].min()
        end = df['datetime'].max()

        if interval == 1:
            label, y, color = 'Kraken 1m (REST)', 4, colors['kraken_1m']
        elif interval == 60:
            label, y, color = 'Kraken 1h (REST)', 3, colors['kraken_1h']
        elif interval == 1440:
            label, y, color = 'Kraken 1d (REST)', 2, colors['kraken_1d']
        else:
            continue

        ax1.barh(y, (end - start).total_seconds() / 86400, left=mdates.date2num(start),
                 height=0.6, color=color, alpha=0.8, label=label)

        # Add text annotation
        mid = start + (end - start) / 2
        ax1.text(mdates.date2num(mid), y, f'{len(df)} records',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Plot WebSocket data
    if not ws_data.empty:
        start = ws_data['datetime'].min()
        end = ws_data['datetime'].max()
        ax1.barh(1, (end - start).total_seconds() / 86400, left=mdates.date2num(start),
                 height=0.6, color=colors['websocket'], alpha=0.8, label='WebSocket')
        mid = start + (end - start) / 2
        ax1.text(mdates.date2num(mid), 1, f'{len(ws_data)} records',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels(['WebSocket', 'Kraken 1d', 'Kraken 1h', 'Kraken 1m'])
    ax1.set_xlim(mdates.date2num(three_months_ago), mdates.date2num(now))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(mdates.date2num(now), color='black', linestyle='--', alpha=0.5, label='Now')

    # =========================================================================
    # Panel 2: Hypothetical Merged Timeline (1h resolution)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title('Hypothetical Merged Data (1h Resolution) - Last 30 Days', fontsize=12, fontweight='bold')

    # Create hourly timeline for last 30 days
    thirty_days_ago = now - timedelta(days=30)
    hours = pd.date_range(start=thirty_days_ago, end=now, freq='h')

    coverage_map = np.zeros(len(hours))  # 0=no data, 1=kraken only, 2=ws only, 3=both

    # Check Kraken 1h coverage
    if 60 in kraken_data:
        kraken_hours = set(kraken_data[60]['datetime'].dt.floor('h'))
        for i, h in enumerate(hours):
            if h in kraken_hours:
                coverage_map[i] = 1

    # Check WebSocket coverage (if we have any)
    if not ws_data.empty:
        ws_hours = set(ws_data['datetime'].dt.floor('h'))
        for i, h in enumerate(hours):
            if h in ws_hours:
                if coverage_map[i] == 1:
                    coverage_map[i] = 3  # Both sources
                else:
                    coverage_map[i] = 2  # WS only

    # Plot as colored bars
    for i, h in enumerate(hours):
        if coverage_map[i] == 0:
            color = '#ecf0f1'  # Light gray - no data
        elif coverage_map[i] == 1:
            color = colors['kraken_1h']  # Blue - Kraken only
        elif coverage_map[i] == 2:
            color = colors['websocket']  # Red - WS only
        else:
            color = colors['overlap']  # Orange - Both
        ax2.axvspan(h, h + timedelta(hours=1), color=color, alpha=0.7)

    ax2.set_xlim(thirty_days_ago, now)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.set_ylabel('Coverage')
    ax2.set_yticks([])

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors['kraken_1h'], label='Kraken REST', alpha=0.7),
        mpatches.Patch(color=colors['websocket'], label='WebSocket', alpha=0.7),
        mpatches.Patch(color=colors['overlap'], label='Both Sources', alpha=0.7),
        mpatches.Patch(color='#ecf0f1', label='No Data', alpha=0.7),
    ]
    ax2.legend(handles=legend_patches, loc='upper left', ncol=4)

    # =========================================================================
    # Panel 3: Data Quality Metrics
    # =========================================================================
    ax3 = axes[2]
    ax3.set_title('Data Quality Metrics by Source', fontsize=12, fontweight='bold')

    sources = []
    metrics_data = {'Completeness': [], 'Consistency': [], 'Freshness': [], 'Overall': []}

    # Kraken 1h quality
    if 60 in kraken_data:
        sources.append('Kraken 1h')
        quality = calculate_data_quality(kraken_data[60])
        metrics_data['Completeness'].append(quality['completeness'])
        metrics_data['Consistency'].append(quality['consistency'])
        metrics_data['Freshness'].append(quality['freshness'])
        metrics_data['Overall'].append(quality['overall'])

    # Kraken 1d quality
    if 1440 in kraken_data:
        sources.append('Kraken 1d')
        quality = calculate_data_quality(kraken_data[1440])
        metrics_data['Completeness'].append(quality['completeness'])
        metrics_data['Consistency'].append(quality['consistency'])
        metrics_data['Freshness'].append(quality['freshness'])
        metrics_data['Overall'].append(quality['overall'])

    # WebSocket quality
    if not ws_data.empty:
        sources.append('WebSocket')
        quality = calculate_data_quality(ws_data)
        metrics_data['Completeness'].append(quality['completeness'])
        metrics_data['Consistency'].append(quality['consistency'])
        metrics_data['Freshness'].append(quality['freshness'])
        metrics_data['Overall'].append(quality['overall'])

    if sources:
        x = np.arange(len(sources))
        width = 0.2

        bars1 = ax3.bar(x - 1.5*width, metrics_data['Completeness'], width, label='Completeness', color='#3498db')
        bars2 = ax3.bar(x - 0.5*width, metrics_data['Consistency'], width, label='Consistency', color='#2ecc71')
        bars3 = ax3.bar(x + 0.5*width, metrics_data['Freshness'], width, label='Freshness', color='#e74c3c')
        bars4 = ax3.bar(x + 1.5*width, metrics_data['Overall'], width, label='Overall', color='#9b59b6')

        ax3.set_xticks(x)
        ax3.set_xticklabels(sources)
        ax3.set_ylabel('Score (%)')
        ax3.set_ylim(0, 110)
        ax3.legend(loc='upper right')
        ax3.axhline(80, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        ax3.axhline(50, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')

        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    # =========================================================================
    # Panel 4: Gap Analysis Summary
    # =========================================================================
    ax4 = axes[3]
    ax4.set_title('Coverage Statistics & Gap Analysis', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Build summary table
    table_data = []
    headers = ['Source', 'Records', 'Duration', 'Coverage %', 'Gaps', 'Largest Gap']

    for interval, df in kraken_data.items():
        if df.empty:
            continue
        if interval == 1:
            name = 'Kraken 1m'
            expected_sec = 60
        elif interval == 60:
            name = 'Kraken 1h'
            expected_sec = 3600
        elif interval == 1440:
            name = 'Kraken 1d'
            expected_sec = 86400
        else:
            continue

        analysis = analyze_coverage(df, expected_sec)
        largest_gap = max([g['duration_seconds'] for g in analysis['gaps']], default=0) / 3600
        table_data.append([
            name,
            f"{analysis['total_records']:,}",
            f"{analysis['duration_hours']:.1f}h",
            f"{analysis['coverage']:.1f}%",
            str(len(analysis['gaps'])),
            f"{largest_gap:.1f}h" if largest_gap > 0 else "N/A"
        ])

    if not ws_data.empty:
        analysis = analyze_coverage(ws_data, 1)  # 1 second expected for ticker
        largest_gap = max([g['duration_seconds'] for g in analysis['gaps']], default=0) / 3600
        table_data.append([
            'WebSocket',
            f"{analysis['total_records']:,}",
            f"{analysis['duration_hours']:.1f}h",
            f"{analysis['coverage']:.1f}%",
            str(len(analysis['gaps'])),
            f"{largest_gap:.1f}h" if largest_gap > 0 else "N/A"
        ])

    if table_data:
        table = ax4.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center',
                         colColours=['#3498db']*len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Color code coverage column
        for i, row in enumerate(table_data):
            coverage = float(row[3].replace('%', ''))
            if coverage >= 95:
                color = '#27ae60'  # Green
            elif coverage >= 80:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            table[(i+1, 3)].set_facecolor(color)
            table[(i+1, 3)].set_text_props(color='white', fontweight='bold')

    # Add merge recommendation text
    ax4.text(0.5, -0.1,
             "Merge Strategy: Use Kraken REST for historical backfill (1h/1d), WebSocket for real-time updates.\n"
             "Gap filling: REST API can fill gaps up to 720 candles back per interval.",
             ha='center', va='top', fontsize=10, style='italic',
             transform=ax4.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {output_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize data merge potential')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--pair', type=str, default='BTC/USD', help='Trading pair')
    parser.add_argument('--output', type=str, default='reports/data_merge_analysis.png', help='Output path')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print(f"Loading data for {args.pair}...")

    # Load Kraken REST data
    kraken_data = load_kraken_rest_data(data_dir, args.pair)
    print(f"Kraken REST intervals available: {list(kraken_data.keys())}")
    for interval, df in kraken_data.items():
        print(f"  {interval}m: {len(df)} records, {df['datetime'].min()} to {df['datetime'].max()}")

    # Load WebSocket data
    ws_data = load_websocket_data(data_dir, args.pair)
    if not ws_data.empty:
        print(f"WebSocket: {len(ws_data)} records, {ws_data['datetime'].min()} to {ws_data['datetime'].max()}")
    else:
        print("WebSocket: No data found")

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    # Create visualization
    create_merge_visualization(kraken_data, ws_data, args.pair, args.output)

if __name__ == "__main__":
    main()
