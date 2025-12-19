#!/usr/bin/env python3
"""
Focused visualization of data merge strategy between Kraken REST and WebSocket.
"""

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

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
            except Exception:
                pass

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['datetime'] = pd.to_datetime(combined['ts'], unit='ms')
    combined = combined.sort_values('datetime').drop_duplicates(subset=['ts'])

    return combined

def create_merge_focus_visualization(kraken_data: dict, ws_data: pd.DataFrame, pair: str, output_path: str):
    """Create focused merge strategy visualization."""

    fig = plt.figure(figsize=(16, 12))

    # Define grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.2], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])  # Full width - main timeline
    ax2 = fig.add_subplot(gs[1, 0])  # Data density
    ax3 = fig.add_subplot(gs[1, 1])  # Gap distribution
    ax4 = fig.add_subplot(gs[2, :])  # Merge result

    fig.suptitle(f'Estrategia de Merge: {pair}\nKraken REST + WebSocket',
                 fontsize=16, fontweight='bold', y=0.98)

    # Colors
    colors = {
        'kraken': '#3498db',      # Blue
        'websocket': '#e74c3c',   # Red
        'merged': '#27ae60',      # Green
        'gap': '#95a5a6',         # Gray
        'overlap': '#f39c12',     # Orange
    }

    now = datetime.now()

    # =========================================================================
    # Panel 1: Main Timeline - Stacked view showing all sources
    # =========================================================================
    ax1.set_title('Cobertura de Datos por Fuente (Últimos 30 Días)', fontsize=12, fontweight='bold')

    thirty_days_ago = now - timedelta(days=30)

    # Create minute-by-minute coverage for detailed view
    # But aggregate to 15-minute blocks for visualization
    block_minutes = 15
    blocks = pd.date_range(start=thirty_days_ago, end=now, freq=f'{block_minutes}min')

    kraken_coverage = np.zeros(len(blocks))
    ws_coverage = np.zeros(len(blocks))

    # Check Kraken 1h coverage (use 1h data as primary)
    if 60 in kraken_data:
        kraken_times = set(kraken_data[60]['datetime'].dt.floor('h'))
        for i, block in enumerate(blocks):
            block_hour = block.floor('h')
            if block_hour in kraken_times:
                kraken_coverage[i] = 1

    # Check WebSocket coverage
    if not ws_data.empty:
        # Group WebSocket data into 15-min blocks
        ws_data_copy = ws_data.copy()
        ws_data_copy['block'] = ws_data_copy['datetime'].dt.floor(f'{block_minutes}min')
        ws_blocks = set(ws_data_copy['block'])
        for i, block in enumerate(blocks):
            if block in ws_blocks:
                ws_coverage[i] = 1

    # Plot as stacked areas
    block_nums = mdates.date2num(blocks)

    # Bottom layer: No data (gray)
    ax1.fill_between(blocks, 0, 1, color=colors['gap'], alpha=0.3, label='Sin Datos')

    # Kraken layer
    ax1.fill_between(blocks, 0, kraken_coverage * 0.5, color=colors['kraken'],
                     alpha=0.8, label='Kraken REST (1h)')

    # WebSocket layer (offset to show on top)
    ax1.fill_between(blocks, 0.5, 0.5 + ws_coverage * 0.5, color=colors['websocket'],
                     alpha=0.8, label='WebSocket')

    # Mark overlap regions
    overlap = (kraken_coverage > 0) & (ws_coverage > 0)
    for i in range(len(blocks) - 1):
        if overlap[i]:
            ax1.axvspan(blocks[i], blocks[i+1], ymin=0.4, ymax=0.6,
                       color=colors['overlap'], alpha=0.9)

    ax1.set_xlim(thirty_days_ago, now)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.25, 0.75])
    ax1.set_yticklabels(['Kraken REST', 'WebSocket'])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax1.legend(loc='upper left', ncol=4)
    ax1.grid(axis='x', alpha=0.3)

    # Add coverage stats as text
    kraken_pct = 100 * kraken_coverage.sum() / len(kraken_coverage)
    ws_pct = 100 * ws_coverage.sum() / len(ws_coverage)
    overlap_pct = 100 * overlap.sum() / len(overlap)

    stats_text = f'Kraken: {kraken_pct:.1f}% | WebSocket: {ws_pct:.1f}% | Overlap: {overlap_pct:.1f}%'
    ax1.text(0.99, 0.02, stats_text, transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # Panel 2: Data Density Over Time
    # =========================================================================
    ax2.set_title('Densidad de Datos por Día', fontsize=11, fontweight='bold')

    # Count records per day
    days = pd.date_range(start=thirty_days_ago.date(), end=now.date(), freq='D')
    kraken_daily = []
    ws_daily = []

    for day in days:
        day_start = pd.Timestamp(day)
        day_end = day_start + timedelta(days=1)

        # Kraken 1h records
        if 60 in kraken_data:
            k_count = ((kraken_data[60]['datetime'] >= day_start) &
                      (kraken_data[60]['datetime'] < day_end)).sum()
        else:
            k_count = 0
        kraken_daily.append(k_count)

        # WebSocket records
        if not ws_data.empty:
            w_count = ((ws_data['datetime'] >= day_start) &
                      (ws_data['datetime'] < day_end)).sum()
        else:
            w_count = 0
        ws_daily.append(w_count)

    x = np.arange(len(days))
    width = 0.35

    bars1 = ax2.bar(x - width/2, kraken_daily, width, label='Kraken REST', color=colors['kraken'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, ws_daily, width, label='WebSocket', color=colors['websocket'], alpha=0.8)

    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Registros')
    ax2.set_xticks(x[::5])
    ax2.set_xticklabels([d.strftime('%m-%d') for d in days[::5]], rotation=45)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, max(max(kraken_daily) if kraken_daily else 0,
                        max(ws_daily) if ws_daily else 0) * 1.2 + 1)

    # =========================================================================
    # Panel 3: Gap Analysis
    # =========================================================================
    ax3.set_title('Distribución de Gaps (WebSocket)', fontsize=11, fontweight='bold')

    if not ws_data.empty and len(ws_data) > 1:
        # Calculate gaps in WebSocket data
        ws_sorted = ws_data.sort_values('datetime')
        gaps = ws_sorted['datetime'].diff().dt.total_seconds() / 60  # in minutes
        gaps = gaps[gaps > 1]  # Only gaps > 1 minute

        if len(gaps) > 0:
            # Categorize gaps
            gap_categories = {
                '1-5 min': ((gaps >= 1) & (gaps < 5)).sum(),
                '5-30 min': ((gaps >= 5) & (gaps < 30)).sum(),
                '30min-1h': ((gaps >= 30) & (gaps < 60)).sum(),
                '1-6h': ((gaps >= 60) & (gaps < 360)).sum(),
                '6-24h': ((gaps >= 360) & (gaps < 1440)).sum(),
                '>24h': (gaps >= 1440).sum(),
            }

            categories = list(gap_categories.keys())
            values = list(gap_categories.values())
            colors_gaps = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#2c3e50']

            bars = ax3.bar(categories, values, color=colors_gaps, alpha=0.8)
            ax3.set_xlabel('Duración del Gap')
            ax3.set_ylabel('Cantidad')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(val), ha='center', va='bottom', fontsize=9)

            # Add total gaps text
            ax3.text(0.98, 0.98, f'Total gaps: {len(gaps)}\nMayor: {gaps.max():.0f} min',
                    transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'Sin datos de WebSocket\npara analizar gaps',
                ha='center', va='center', fontsize=12, color='gray')
        ax3.set_xticks([])
        ax3.set_yticks([])

    # =========================================================================
    # Panel 4: Merged Result Preview
    # =========================================================================
    ax4.set_title('Resultado del Merge: Cobertura Combinada', fontsize=12, fontweight='bold')

    # Show what merged data would look like
    merged_coverage = np.maximum(kraken_coverage, ws_coverage)

    # Create color-coded timeline
    for i in range(len(blocks) - 1):
        if kraken_coverage[i] > 0 and ws_coverage[i] > 0:
            color = colors['overlap']  # Both sources
            label = 'Ambas fuentes'
        elif kraken_coverage[i] > 0:
            color = colors['kraken']  # Kraken only
            label = 'Solo Kraken'
        elif ws_coverage[i] > 0:
            color = colors['websocket']  # WS only
            label = 'Solo WebSocket'
        else:
            color = colors['gap']  # No data
            label = 'Sin datos'

        ax4.axvspan(blocks[i], blocks[i+1], color=color, alpha=0.8)

    ax4.set_xlim(thirty_days_ago, now)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=3))

    # Custom legend
    legend_patches = [
        mpatches.Patch(color=colors['kraken'], label='Solo Kraken REST', alpha=0.8),
        mpatches.Patch(color=colors['websocket'], label='Solo WebSocket', alpha=0.8),
        mpatches.Patch(color=colors['overlap'], label='Ambas Fuentes', alpha=0.8),
        mpatches.Patch(color=colors['gap'], label='Sin Datos (Gap)', alpha=0.8),
    ]
    ax4.legend(handles=legend_patches, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.15))

    # Calculate merged coverage
    merged_pct = 100 * merged_coverage.sum() / len(merged_coverage)
    gap_pct = 100 - merged_pct

    # Add summary box
    summary_text = (
        f"RESUMEN DEL MERGE\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Cobertura Kraken:    {kraken_pct:>6.1f}%\n"
        f"Cobertura WebSocket: {ws_pct:>6.1f}%\n"
        f"Overlap (ambas):     {overlap_pct:>6.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Cobertura Merged:    {merged_pct:>6.1f}%\n"
        f"Gaps restantes:      {gap_pct:>6.1f}%"
    )

    ax4.text(1.02, 0.5, summary_text, transform=ax4.transAxes, ha='left', va='center',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.9))

    # Add recommendation
    if gap_pct > 20:
        recommendation = "Recomendación: Capturar más datos históricos con REST API"
        rec_color = '#e74c3c'
    elif gap_pct > 5:
        recommendation = "Recomendación: El merge cubre la mayoría, algunos gaps menores"
        rec_color = '#f39c12'
    else:
        recommendation = "Excelente cobertura con el merge de ambas fuentes"
        rec_color = '#27ae60'

    fig.text(0.5, 0.02, recommendation, ha='center', fontsize=11, fontweight='bold', color=rec_color)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Guardado: {output_path}")

    # Print summary to console
    print(f"\n{'='*50}")
    print(f"RESUMEN DE MERGE - {pair}")
    print(f"{'='*50}")
    print(f"Kraken REST (1h): {kraken_pct:.1f}% cobertura")
    print(f"WebSocket:        {ws_pct:.1f}% cobertura")
    print(f"Overlap:          {overlap_pct:.1f}%")
    print(f"{'='*50}")
    print(f"Merged total:     {merged_pct:.1f}% cobertura")
    print(f"Gaps restantes:   {gap_pct:.1f}%")
    print(f"{'='*50}")

def main():
    data_dir = Path('data')
    pair = 'BTC/USD'
    output_path = 'reports/merge_strategy.png'

    print(f"Cargando datos para {pair}...")

    kraken_data = load_kraken_rest_data(data_dir, pair)
    ws_data = load_websocket_data(data_dir, pair)

    for interval, df in kraken_data.items():
        print(f"  Kraken {interval}m: {len(df)} registros")

    if not ws_data.empty:
        print(f"  WebSocket: {len(ws_data)} registros")

    Path(output_path).parent.mkdir(exist_ok=True)
    create_merge_focus_visualization(kraken_data, ws_data, pair, output_path)

if __name__ == "__main__":
    main()
