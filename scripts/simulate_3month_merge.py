#!/usr/bin/env python3
"""
Simulate 3-month data merge scenario between Kraken REST and WebSocket.

Shows hypothetical coverage with continuous WebSocket operation.
"""

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import random

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

def simulate_websocket_coverage(start_date: datetime, end_date: datetime,
                                uptime_pct: float = 0.85,
                                avg_session_hours: float = 8,
                                avg_gap_hours: float = 2) -> list:
    """
    Simulate realistic WebSocket coverage with gaps.

    Args:
        start_date: Start of simulation
        end_date: End of simulation
        uptime_pct: Target uptime percentage (0-1)
        avg_session_hours: Average session duration before restart
        avg_gap_hours: Average gap duration

    Returns:
        List of (start, end) tuples representing active sessions
    """
    sessions = []
    current_time = start_date

    while current_time < end_date:
        # Session duration (exponential distribution around average)
        session_hours = np.random.exponential(avg_session_hours)
        session_hours = max(0.5, min(session_hours, 48))  # Clamp between 30min and 48h

        session_end = current_time + timedelta(hours=session_hours)
        if session_end > end_date:
            session_end = end_date

        sessions.append((current_time, session_end))

        # Gap duration (exponential distribution)
        gap_hours = np.random.exponential(avg_gap_hours)
        gap_hours = max(0.1, min(gap_hours, 24))  # Clamp between 6min and 24h

        current_time = session_end + timedelta(hours=gap_hours)

    return sessions

def create_3month_simulation(kraken_data: dict, pair: str, output_path: str):
    """Create 3-month merge simulation visualization."""

    np.random.seed(42)  # For reproducibility

    fig = plt.figure(figsize=(18, 14))

    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1.2], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])  # Timeline overview
    ax2 = fig.add_subplot(gs[1, 0])  # Weekly coverage comparison
    ax3 = fig.add_subplot(gs[1, 1])  # Data source breakdown
    ax4 = fig.add_subplot(gs[2, :])  # Detailed daily view
    ax5 = fig.add_subplot(gs[3, :])  # Merged result

    fig.suptitle(f'Simulación de Merge 3 Meses: {pair}\nEscenario Hipotético con WebSocket Continuo',
                 fontsize=16, fontweight='bold', y=0.98)

    colors = {
        'kraken_1h': '#3498db',
        'kraken_1d': '#9b59b6',
        'websocket': '#e74c3c',
        'merged': '#27ae60',
        'gap': '#bdc3c7',
        'overlap': '#f39c12',
    }

    now = datetime.now()
    three_months_ago = now - timedelta(days=90)

    # Generate simulated WebSocket sessions (85% uptime scenario)
    ws_sessions_85 = simulate_websocket_coverage(three_months_ago, now,
                                                  uptime_pct=0.85,
                                                  avg_session_hours=12,
                                                  avg_gap_hours=2)

    # Also simulate a poor scenario (50% uptime) for comparison
    ws_sessions_50 = simulate_websocket_coverage(three_months_ago, now,
                                                  uptime_pct=0.50,
                                                  avg_session_hours=4,
                                                  avg_gap_hours=4)

    # Create hourly timeline
    hours = pd.date_range(start=three_months_ago, end=now, freq='h')

    # Kraken 1h coverage (real data - only last 30 days)
    kraken_1h_coverage = np.zeros(len(hours))
    if 60 in kraken_data:
        kraken_hours = set(kraken_data[60]['datetime'].dt.floor('h'))
        for i, h in enumerate(hours):
            if h in kraken_hours:
                kraken_1h_coverage[i] = 1

    # Kraken 1d coverage (real data - 2 years, so full coverage)
    kraken_1d_coverage = np.zeros(len(hours))
    if 1440 in kraken_data:
        kraken_days = set(kraken_data[1440]['datetime'].dt.floor('D'))
        for i, h in enumerate(hours):
            if h.floor('D') in kraken_days:
                kraken_1d_coverage[i] = 1

    # WebSocket coverage (simulated 85% uptime)
    ws_coverage_85 = np.zeros(len(hours))
    for session_start, session_end in ws_sessions_85:
        for i, h in enumerate(hours):
            if session_start <= h < session_end:
                ws_coverage_85[i] = 1

    # WebSocket coverage (simulated 50% uptime)
    ws_coverage_50 = np.zeros(len(hours))
    for session_start, session_end in ws_sessions_50:
        for i, h in enumerate(hours):
            if session_start <= h < session_end:
                ws_coverage_50[i] = 1

    # =========================================================================
    # Panel 1: 3-Month Timeline Overview
    # =========================================================================
    ax1.set_title('Cobertura de 3 Meses - Vista General', fontsize=12, fontweight='bold')

    # Plot as stacked horizontal bars
    y_positions = [4, 3, 2, 1]
    labels = ['Kraken 1d (Real)', 'Kraken 1h (Real)', 'WebSocket (Sim 85%)', 'WebSocket (Sim 50%)']
    coverages = [kraken_1d_coverage, kraken_1h_coverage, ws_coverage_85, ws_coverage_50]
    bar_colors = [colors['kraken_1d'], colors['kraken_1h'], colors['websocket'], '#c0392b']

    for y, coverage, color in zip(y_positions, coverages, bar_colors):
        # Find continuous segments
        segments = []
        start_idx = None
        for i, val in enumerate(coverage):
            if val > 0 and start_idx is None:
                start_idx = i
            elif val == 0 and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(coverage)))

        for seg_start, seg_end in segments:
            ax1.barh(y, mdates.date2num(hours[seg_end-1]) - mdates.date2num(hours[seg_start]),
                    left=mdates.date2num(hours[seg_start]), height=0.6, color=color, alpha=0.8)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(mdates.date2num(three_months_ago), mdates.date2num(now))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax1.grid(axis='x', alpha=0.3)

    # Add coverage percentages
    for y, coverage, label in zip(y_positions, coverages, labels):
        pct = 100 * coverage.sum() / len(coverage)
        ax1.text(mdates.date2num(now) + 1, y, f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')

    # =========================================================================
    # Panel 2: Weekly Coverage Comparison
    # =========================================================================
    ax2.set_title('Cobertura Semanal Comparativa', fontsize=11, fontweight='bold')

    weeks = pd.date_range(start=three_months_ago, end=now, freq='W')
    week_labels = [w.strftime('%m-%d') for w in weeks]

    kraken_weekly = []
    ws_85_weekly = []
    ws_50_weekly = []

    for i, week_start in enumerate(weeks[:-1]):
        week_end = weeks[i+1]
        mask = (hours >= week_start) & (hours < week_end)

        kraken_weekly.append(100 * kraken_1h_coverage[mask].mean() if mask.sum() > 0 else 0)
        ws_85_weekly.append(100 * ws_coverage_85[mask].mean() if mask.sum() > 0 else 0)
        ws_50_weekly.append(100 * ws_coverage_50[mask].mean() if mask.sum() > 0 else 0)

    x = np.arange(len(weeks) - 1)
    width = 0.25

    ax2.bar(x - width, kraken_weekly, width, label='Kraken 1h', color=colors['kraken_1h'], alpha=0.8)
    ax2.bar(x, ws_85_weekly, width, label='WS 85%', color=colors['websocket'], alpha=0.8)
    ax2.bar(x + width, ws_50_weekly, width, label='WS 50%', color='#c0392b', alpha=0.8)

    ax2.set_xlabel('Semana')
    ax2.set_ylabel('Cobertura %')
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(week_labels[:-1:2], rotation=45)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color='green', linestyle='--', alpha=0.5)

    # =========================================================================
    # Panel 3: Data Source Breakdown (Pie charts)
    # =========================================================================
    ax3.set_title('Distribución de Cobertura por Escenario', fontsize=11, fontweight='bold')

    # Calculate merged coverages
    merged_85 = np.maximum(kraken_1h_coverage, ws_coverage_85)
    merged_50 = np.maximum(kraken_1h_coverage, ws_coverage_50)

    # For 85% scenario
    only_kraken_85 = ((kraken_1h_coverage > 0) & (ws_coverage_85 == 0)).sum()
    only_ws_85 = ((kraken_1h_coverage == 0) & (ws_coverage_85 > 0)).sum()
    both_85 = ((kraken_1h_coverage > 0) & (ws_coverage_85 > 0)).sum()
    none_85 = ((kraken_1h_coverage == 0) & (ws_coverage_85 == 0)).sum()

    # Create grouped bar chart instead of pie
    scenarios = ['Escenario 85%\n(Óptimo)', 'Escenario 50%\n(Actual Fly.io)']

    only_ws_50 = ((kraken_1h_coverage == 0) & (ws_coverage_50 > 0)).sum()
    only_kraken_50 = ((kraken_1h_coverage > 0) & (ws_coverage_50 == 0)).sum()
    both_50 = ((kraken_1h_coverage > 0) & (ws_coverage_50 > 0)).sum()
    none_50 = ((kraken_1h_coverage == 0) & (ws_coverage_50 == 0)).sum()

    total = len(hours)
    data_85 = [100*only_kraken_85/total, 100*only_ws_85/total, 100*both_85/total, 100*none_85/total]
    data_50 = [100*only_kraken_50/total, 100*only_ws_50/total, 100*both_50/total, 100*none_50/total]

    x = np.arange(4)
    width = 0.35

    bars1 = ax3.bar(x - width/2, data_85, width, label='85% Uptime', color=colors['merged'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, data_50, width, label='50% Uptime', color='#95a5a6', alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(['Solo\nKraken', 'Solo\nWebSocket', 'Ambas\nFuentes', 'Sin\nDatos'])
    ax3.set_ylabel('% del Tiempo')
    ax3.legend()

    for bar in bars1:
        h = bar.get_height()
        if h > 2:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}%', ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 2:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}%', ha='center', fontsize=8)

    # =========================================================================
    # Panel 4: Detailed Daily View (Last 30 days)
    # =========================================================================
    ax4.set_title('Vista Detallada: Últimos 30 Días (Resolución Horaria)', fontsize=11, fontweight='bold')

    thirty_days_ago = now - timedelta(days=30)
    mask_30d = hours >= thirty_days_ago
    hours_30d = hours[mask_30d]

    # Create color-coded timeline
    for i, h in enumerate(hours_30d[:-1]):
        idx = np.where(hours == h)[0][0]

        has_kraken = kraken_1h_coverage[idx] > 0
        has_ws = ws_coverage_85[idx] > 0

        if has_kraken and has_ws:
            color = colors['overlap']
        elif has_kraken:
            color = colors['kraken_1h']
        elif has_ws:
            color = colors['websocket']
        else:
            color = colors['gap']

        ax4.axvspan(h, hours_30d[i+1], color=color, alpha=0.8)

    ax4.set_xlim(thirty_days_ago, now)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=3))

    legend_patches = [
        mpatches.Patch(color=colors['kraken_1h'], label='Solo Kraken', alpha=0.8),
        mpatches.Patch(color=colors['websocket'], label='Solo WebSocket', alpha=0.8),
        mpatches.Patch(color=colors['overlap'], label='Ambas Fuentes', alpha=0.8),
        mpatches.Patch(color=colors['gap'], label='Sin Datos', alpha=0.8),
    ]
    ax4.legend(handles=legend_patches, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))

    # =========================================================================
    # Panel 5: Merged Result Summary
    # =========================================================================
    ax5.set_title('Resultado Final del Merge (3 Meses)', fontsize=12, fontweight='bold')

    # Show merged timeline for full 3 months
    for i, h in enumerate(hours[:-1]):
        has_kraken = kraken_1h_coverage[i] > 0 or kraken_1d_coverage[i] > 0
        has_ws = ws_coverage_85[i] > 0

        if has_kraken and has_ws:
            color = colors['overlap']
        elif has_kraken:
            color = colors['kraken_1h']
        elif has_ws:
            color = colors['websocket']
        else:
            color = colors['gap']

        ax5.axvspan(h, hours[i+1], color=color, alpha=0.8)

    ax5.set_xlim(three_months_ago, now)
    ax5.set_ylim(0, 1)
    ax5.set_yticks([])
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax5.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    # Calculate final stats
    merged_full = np.maximum(np.maximum(kraken_1h_coverage, kraken_1d_coverage), ws_coverage_85)
    merged_pct = 100 * merged_full.sum() / len(merged_full)
    gap_pct = 100 - merged_pct

    kraken_only = ((kraken_1h_coverage > 0) | (kraken_1d_coverage > 0)) & (ws_coverage_85 == 0)
    ws_only = (ws_coverage_85 > 0) & (kraken_1h_coverage == 0) & (kraken_1d_coverage == 0)
    overlap = ((kraken_1h_coverage > 0) | (kraken_1d_coverage > 0)) & (ws_coverage_85 > 0)

    # Summary box
    summary_text = (
        f"RESUMEN MERGE 3 MESES\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Kraken 1h (real):     {100*kraken_1h_coverage.sum()/len(hours):>6.1f}%\n"
        f"Kraken 1d (real):     {100*kraken_1d_coverage.sum()/len(hours):>6.1f}%\n"
        f"WebSocket (sim 85%):  {100*ws_coverage_85.sum()/len(hours):>6.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Solo Kraken:          {100*kraken_only.sum()/len(hours):>6.1f}%\n"
        f"Solo WebSocket:       {100*ws_only.sum()/len(hours):>6.1f}%\n"
        f"Ambas fuentes:        {100*overlap.sum()/len(hours):>6.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"COBERTURA TOTAL:      {merged_pct:>6.1f}%\n"
        f"Gaps restantes:       {gap_pct:>6.1f}%"
    )

    ax5.text(1.02, 0.5, summary_text, transform=ax5.transAxes, ha='left', va='center',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#27ae60', linewidth=2))

    # Recommendation
    if merged_pct >= 99:
        rec = "EXCELENTE: Cobertura casi perfecta con el merge"
        rec_color = '#27ae60'
    elif merged_pct >= 90:
        rec = "MUY BUENO: Cobertura sólida, gaps mínimos"
        rec_color = '#27ae60'
    elif merged_pct >= 80:
        rec = "BUENO: Cobertura aceptable para ML"
        rec_color = '#f39c12'
    else:
        rec = "MEJORABLE: Considerar más fuentes de datos"
        rec_color = '#e74c3c'

    fig.text(0.5, 0.01, rec, ha='center', fontsize=12, fontweight='bold', color=rec_color)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Guardado: {output_path}")

    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"SIMULACIÓN DE MERGE 3 MESES - {pair}")
    print(f"{'='*60}")
    print(f"\nDatos Reales:")
    print(f"  Kraken 1h: {100*kraken_1h_coverage.sum()/len(hours):.1f}% (últimos 30 días)")
    print(f"  Kraken 1d: {100*kraken_1d_coverage.sum()/len(hours):.1f}% (2 años)")
    print(f"\nDatos Simulados (WebSocket):")
    print(f"  Escenario 85% uptime: {100*ws_coverage_85.sum()/len(hours):.1f}%")
    print(f"  Escenario 50% uptime: {100*ws_coverage_50.sum()/len(hours):.1f}%")
    print(f"\nResultado del Merge (Kraken + WS 85%):")
    print(f"  Cobertura total: {merged_pct:.1f}%")
    print(f"  Gaps restantes:  {gap_pct:.1f}%")
    print(f"\nDesglose:")
    print(f"  Solo Kraken:    {100*kraken_only.sum()/len(hours):.1f}%")
    print(f"  Solo WebSocket: {100*ws_only.sum()/len(hours):.1f}%")
    print(f"  Ambas fuentes:  {100*overlap.sum()/len(hours):.1f}%")
    print(f"{'='*60}")

def main():
    data_dir = Path('data')
    pair = 'BTC/USD'
    output_path = 'reports/simulation_3month_merge.png'

    print(f"Cargando datos reales para {pair}...")
    kraken_data = load_kraken_rest_data(data_dir, pair)

    for interval, df in kraken_data.items():
        print(f"  Kraken {interval}m: {len(df)} registros")

    print("\nGenerando simulación de 3 meses...")
    Path(output_path).parent.mkdir(exist_ok=True)
    create_3month_simulation(kraken_data, pair, output_path)

if __name__ == "__main__":
    main()
