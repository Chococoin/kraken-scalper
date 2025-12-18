#!/usr/bin/env python3
"""
Breakout Post-Range Strategy Grid

Hypothesis: After a big drop, price consolidates in a range.
Strategy: Enter LONG on upside breakout from consolidation range.

Flow:
1. Detect shock (multi-bar drop ≥ threshold)
2. Wait for consolidation (N bars with range < X%)
3. Enter LONG when price breaks above consolidation high
4. TP/SL based on consolidation range size

Costs: Taker (0.4%) + slippage (0.05%) = 0.45% per side, ~0.9% round-trip
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Taker costs (conservative)
TAKER_FEE = 0.004  # 0.4%
SLIPPAGE = 0.0005  # 0.05%
COSTS_PER_SIDE = TAKER_FEE + SLIPPAGE  # 0.45%
ROUND_TRIP_COSTS = COSTS_PER_SIDE * 2  # 0.9%


@dataclass
class BreakoutConfig:
    name: str
    # Shock detection
    shock_lookback: int      # bars to measure drop
    shock_threshold: float   # minimum drop % (e.g., 0.012 = 1.2%)
    # Consolidation detection
    consol_bars: int         # bars to wait for consolidation
    consol_max_range: float  # max range % to qualify as consolidation
    # Entry
    breakout_buffer: float   # % above consol high to confirm breakout
    # Exit
    tp_mult: float           # TP = breakout + consol_range * tp_mult
    sl_mult: float           # SL = breakout - consol_range * sl_mult
    max_hold: int            # max bars to hold

    @property
    def breakeven_rate(self) -> float:
        """Win rate needed to break even (without costs)"""
        return self.sl_mult / (self.tp_mult + self.sl_mult)


# Grid of breakout configurations
GRID: List[BreakoutConfig] = [
    # === Quick breakouts (short consolidation) ===
    BreakoutConfig("quick_tight", shock_lookback=3, shock_threshold=0.012,
                   consol_bars=5, consol_max_range=0.004, breakout_buffer=0.001,
                   tp_mult=1.5, sl_mult=1.0, max_hold=15),
    BreakoutConfig("quick_wide", shock_lookback=3, shock_threshold=0.012,
                   consol_bars=5, consol_max_range=0.006, breakout_buffer=0.001,
                   tp_mult=2.0, sl_mult=1.0, max_hold=20),

    # === Standard breakouts (medium consolidation) ===
    BreakoutConfig("std_conservative", shock_lookback=3, shock_threshold=0.012,
                   consol_bars=10, consol_max_range=0.005, breakout_buffer=0.001,
                   tp_mult=1.5, sl_mult=0.75, max_hold=30),
    BreakoutConfig("std_balanced", shock_lookback=3, shock_threshold=0.012,
                   consol_bars=10, consol_max_range=0.005, breakout_buffer=0.001,
                   tp_mult=2.0, sl_mult=1.0, max_hold=30),
    BreakoutConfig("std_aggressive", shock_lookback=3, shock_threshold=0.012,
                   consol_bars=10, consol_max_range=0.005, breakout_buffer=0.001,
                   tp_mult=3.0, sl_mult=1.0, max_hold=45),

    # === 5-bar shock (larger drops) ===
    BreakoutConfig("5bar_quick", shock_lookback=5, shock_threshold=0.012,
                   consol_bars=5, consol_max_range=0.004, breakout_buffer=0.001,
                   tp_mult=1.5, sl_mult=1.0, max_hold=15),
    BreakoutConfig("5bar_std", shock_lookback=5, shock_threshold=0.012,
                   consol_bars=10, consol_max_range=0.005, breakout_buffer=0.001,
                   tp_mult=2.0, sl_mult=1.0, max_hold=30),
    BreakoutConfig("5bar_patient", shock_lookback=5, shock_threshold=0.015,
                   consol_bars=15, consol_max_range=0.006, breakout_buffer=0.001,
                   tp_mult=2.5, sl_mult=1.0, max_hold=45),

    # === Higher threshold (rarer, bigger moves) ===
    BreakoutConfig("rare_quick", shock_lookback=3, shock_threshold=0.015,
                   consol_bars=5, consol_max_range=0.005, breakout_buffer=0.001,
                   tp_mult=2.0, sl_mult=1.0, max_hold=20),
    BreakoutConfig("rare_std", shock_lookback=5, shock_threshold=0.015,
                   consol_bars=10, consol_max_range=0.006, breakout_buffer=0.001,
                   tp_mult=2.5, sl_mult=1.0, max_hold=30),

    # === Extended consolidation (stronger signal) ===
    BreakoutConfig("ext_tight", shock_lookback=3, shock_threshold=0.012,
                   consol_bars=15, consol_max_range=0.004, breakout_buffer=0.001,
                   tp_mult=2.0, sl_mult=0.75, max_hold=30),
    BreakoutConfig("ext_wide", shock_lookback=5, shock_threshold=0.012,
                   consol_bars=15, consol_max_range=0.006, breakout_buffer=0.001,
                   tp_mult=2.5, sl_mult=1.0, max_hold=45),
]


def detect_shocks(df: pd.DataFrame, lookback: int, threshold: float) -> pd.Series:
    """Detect bars where price dropped by threshold over lookback period."""
    if lookback == 1:
        returns = df['close'].pct_change()
    else:
        returns = df['close'] / df['close'].shift(lookback) - 1
    return returns < -threshold


def find_consolidation(df: pd.DataFrame, start_pos: int, consol_bars: int,
                       max_range_pct: float) -> Optional[Tuple[float, float, int]]:
    """
    Look for consolidation starting at start_pos.

    Returns: (consol_high, consol_low, end_pos) or None if no consolidation found.
    """
    if start_pos + consol_bars >= len(df):
        return None

    window = df.iloc[start_pos:start_pos + consol_bars]
    consol_high = window['high'].max()
    consol_low = window['low'].min()
    consol_range = (consol_high - consol_low) / window.iloc[0]['open']

    if consol_range <= max_range_pct:
        return (consol_high, consol_low, start_pos + consol_bars)

    return None


def find_breakout(df: pd.DataFrame, start_pos: int, consol_high: float,
                  buffer: float, max_wait: int = 10) -> Optional[int]:
    """
    Look for upside breakout above consol_high + buffer.

    Returns: position of breakout bar, or None.
    """
    breakout_level = consol_high * (1 + buffer)

    for i in range(start_pos, min(start_pos + max_wait, len(df))):
        if df.iloc[i]['high'] >= breakout_level:
            return i

    return None


def evaluate_breakout_trade(df: pd.DataFrame, entry_pos: int, entry_price: float,
                            consol_range: float, tp_mult: float, sl_mult: float,
                            max_hold: int, costs: float) -> Tuple[str, float, int]:
    """
    Evaluate a breakout trade.

    Returns: (outcome, pnl, bars_held)
    """
    tp_price = entry_price + consol_range * tp_mult
    sl_price = entry_price - consol_range * sl_mult

    for i in range(entry_pos, min(entry_pos + max_hold, len(df))):
        bar = df.iloc[i]

        # Check TP (using high)
        if bar['high'] >= tp_price:
            pnl = (tp_price / entry_price - 1) - costs
            return ('TP', pnl, i - entry_pos + 1)

        # Check SL (using low)
        if bar['low'] <= sl_price:
            pnl = (sl_price / entry_price - 1) - costs
            return ('SL', pnl, i - entry_pos + 1)

    # Expired
    if entry_pos + max_hold - 1 < len(df):
        exit_price = df.iloc[entry_pos + max_hold - 1]['close']
        pnl = (exit_price / entry_price - 1) - costs
        return ('EXP', pnl, max_hold)

    return ('SKIP', 0.0, 0)


def run_config(df: pd.DataFrame, config: BreakoutConfig, costs: float = ROUND_TRIP_COSTS) -> dict:
    """Run backtest for a single breakout configuration."""

    # Detect shocks
    shocks = detect_shocks(df, config.shock_lookback, config.shock_threshold)
    shock_indices = df.index[shocks].tolist()

    results = {
        'name': config.name,
        'config': config,
        'shocks': len(shock_indices),
        'consolidations': 0,
        'breakouts': 0,
        'tp_count': 0,
        'sl_count': 0,
        'exp_count': 0,
        'total_pnl': 0.0,
        'trades': []
    }

    for idx in shock_indices:
        pos = df.index.get_loc(idx)
        consol_start = pos + 1

        # Look for consolidation
        consol = find_consolidation(df, consol_start, config.consol_bars, config.consol_max_range)
        if consol is None:
            continue

        consol_high, consol_low, consol_end = consol
        consol_range = consol_high - consol_low
        results['consolidations'] += 1

        # Look for breakout
        breakout_pos = find_breakout(df, consol_end, consol_high, config.breakout_buffer)
        if breakout_pos is None:
            continue

        results['breakouts'] += 1

        # Enter at breakout bar's close (conservative entry)
        entry_price = df.iloc[breakout_pos]['close']

        # Evaluate trade
        outcome, pnl, bars = evaluate_breakout_trade(
            df, breakout_pos + 1, entry_price, consol_range,
            config.tp_mult, config.sl_mult, config.max_hold, costs
        )

        if outcome == 'TP':
            results['tp_count'] += 1
        elif outcome == 'SL':
            results['sl_count'] += 1
        elif outcome == 'EXP':
            results['exp_count'] += 1
        else:
            continue

        results['total_pnl'] += pnl
        results['trades'].append({
            'shock_idx': idx,
            'entry_price': entry_price,
            'outcome': outcome,
            'pnl': pnl,
            'bars': bars
        })

    # Calculate metrics
    total_trades = results['tp_count'] + results['sl_count'] + results['exp_count']
    if total_trades > 0:
        results['win_rate'] = results['tp_count'] / total_trades
        results['avg_pnl'] = results['total_pnl'] / total_trades
    else:
        results['win_rate'] = 0
        results['avg_pnl'] = 0

    return results


def run_grid(df: pd.DataFrame, configs: List[BreakoutConfig] = GRID,
             costs: float = ROUND_TRIP_COSTS) -> pd.DataFrame:
    """Run all configurations and return results DataFrame."""
    all_results = []

    for config in configs:
        result = run_config(df, config, costs)
        total = result['tp_count'] + result['sl_count'] + result['exp_count']

        p_win = result['tp_count'] / total if total > 0 else 0
        p_loss = result['sl_count'] / total if total > 0 else 0

        all_results.append({
            'name': result['name'],
            'shock_lb': config.shock_lookback,
            'shock_th': f"{config.shock_threshold*100:.1f}%",
            'consol': f"{config.consol_bars}b/{config.consol_max_range*100:.1f}%",
            'tp/sl': f"{config.tp_mult:.1f}/{config.sl_mult:.1f}",
            'hold': config.max_hold,
            'BE': f"{config.breakeven_rate*100:.0f}%",
            'shocks': result['shocks'],
            'consol': result['consolidations'],
            'brk': result['breakouts'],
            'n': total,
            'p_win': f"{p_win*100:.0f}%" if total > 0 else '-',
            'p_loss': f"{p_loss*100:.0f}%" if total > 0 else '-',
            'EV': f"{result['avg_pnl']*100:.2f}%" if total > 0 else '-',
            'total': f"{result['total_pnl']*100:.2f}%",
        })

    return pd.DataFrame(all_results)


def load_data(pair: str = "BTC/USD", interval: str = "1min") -> Optional[pd.DataFrame]:
    """Load trade data and convert to OHLC bars."""
    data_dir = Path("data/crypto/trade")

    if not data_dir.exists():
        return None

    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df = df[df['pair'] == pair]
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        return None

    trades = pd.concat(dfs, ignore_index=True)
    trades['timestamp'] = pd.to_datetime(trades['ts'], unit='ms')
    trades = trades.set_index('timestamp').sort_index()

    # Build OHLC
    ohlc = trades['price'].resample(interval).ohlc()
    volume = trades['qty'].resample(interval).sum()
    ohlc['volume'] = volume
    ohlc = ohlc.ffill().dropna()

    return ohlc


def main():
    """Run breakout grid analysis."""
    print("=" * 80)
    print("BREAKOUT POST-RANGE STRATEGY GRID")
    print("=" * 80)
    print(f"Costs: Taker {TAKER_FEE*100:.1f}% + Slippage {SLIPPAGE*100:.2f}% = {ROUND_TRIP_COSTS*100:.1f}% round-trip")
    print()

    # Load data
    print("Loading data...")
    df = load_data("BTC/USD", "1min")

    if df is None or len(df) == 0:
        print("No data available.")
        return

    hours = len(df) / 60
    print(f"Loaded {len(df)} bars (~{hours:.1f} hours)")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print()

    # Run grid
    print("Running breakout grid...")
    print()

    results = run_grid(df, GRID)

    # Sort by total P&L
    results['pnl_sort'] = results['total'].str.rstrip('%').astype(float)
    results = results.sort_values('pnl_sort', ascending=False)

    print("RESULTS (sorted by P&L):")
    print("-" * 120)
    print(results.drop(columns=['pnl_sort']).to_string(index=False))
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    positive = results[results['pnl_sort'] > 0]
    if len(positive) > 0:
        print(f"\n✅ Configs with positive P&L: {len(positive)}")
        for _, row in positive.iterrows():
            print(f"  - {row['name']}: {row['total']} ({row['n']} trades)")
    else:
        print("\n❌ No configs with positive P&L")

    # Funnel analysis
    print("\nFUNNEL ANALYSIS:")
    total_shocks = results['shocks'].sum()
    total_consol = results['consol'].sum()
    total_brk = results['brk'].sum()
    total_trades = results['n'].sum()

    print(f"  Shocks detected: {total_shocks}")
    print(f"  → Consolidations: {total_consol}")
    print(f"  → Breakouts: {total_brk}")
    print(f"  → Trades executed: {total_trades}")

    if total_shocks > 0:
        print(f"\n  Conversion rates:")
        print(f"    Shock → Consolidation: {total_consol/total_shocks*100:.0f}%")
        if total_consol > 0:
            print(f"    Consolidation → Breakout: {total_brk/total_consol*100:.0f}%")


if __name__ == "__main__":
    main()
