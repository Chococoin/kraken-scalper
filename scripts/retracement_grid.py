#!/usr/bin/env python3
"""
Retracement Strategy Grid Search

Tests multiple configurations of the retracement strategy to find
combinations that might have edge.

Grid dimensions:
- Lookback: bars to detect drop (1, 3, 5)
- Drop threshold: minimum drop % to trigger (0.3%, 0.5%, 1.0%)
- TP target: % of drop to recover (30%, 50%, 75%, 100%)
- SL multiplier: % of drop for stop loss (25%, 50%, 75%)
- Horizon: max bars to hold (5, 10, 20, 30)

Constraints:
- TP% < SL% avoided (negative expectancy)
- Total configs: ~20 carefully selected
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

# Configuration for a single test
@dataclass
class RetrConfig:
    name: str
    lookback: int        # bars to measure drop
    drop_threshold: float  # minimum drop % (e.g., 0.005 = 0.5%)
    tp_pct: float        # TP as % of drop (e.g., 0.5 = 50%)
    sl_pct: float        # SL as % of drop (e.g., 0.75 = 75%)
    horizon: int         # max bars to hold

    @property
    def breakeven_rate(self) -> float:
        """Breakeven win rate (without costs)"""
        return self.sl_pct / (self.tp_pct + self.sl_pct)

    @property
    def breakeven_rate_with_costs(self, costs: float = 0.005) -> float:
        """Breakeven win rate with round-trip costs"""
        # BE = (SL + costs) / (TP + SL)
        # But TP and SL are in terms of drop, need to convert
        # If drop is D, TP = D * tp_pct, SL = D * sl_pct
        # Assume average drop is ~0.5%, so:
        avg_drop = 0.005
        tp_abs = avg_drop * self.tp_pct
        sl_abs = avg_drop * self.sl_pct
        return (sl_abs + costs) / (tp_abs + sl_abs)


# ChatGPT's systematic grid of 18 configurations
# Organized by "family":
#   - Frecuentes (0.3-0.5%): micro-rebounds, high cost sensitivity
#   - Intermedias (0.8-1.0%): core plausible range
#   - Raras (1.5%): few trades, edge appears here first if it exists
GRID: List[RetrConfig] = [
    # === Lookback 1 (single bar) ===
    # Frecuentes
    RetrConfig("lb1_th0p3_tp25_sl25", lookback=1, drop_threshold=0.003, tp_pct=0.25, sl_pct=0.25, horizon=10),
    RetrConfig("lb1_th0p3_tp30_sl40", lookback=1, drop_threshold=0.003, tp_pct=0.30, sl_pct=0.40, horizon=10),
    RetrConfig("lb1_th0p5_tp30_sl25", lookback=1, drop_threshold=0.005, tp_pct=0.30, sl_pct=0.25, horizon=15),
    RetrConfig("lb1_th0p5_tp50_sl40", lookback=1, drop_threshold=0.005, tp_pct=0.50, sl_pct=0.40, horizon=15),
    # Intermedias
    RetrConfig("lb1_th0p8_tp50_sl25", lookback=1, drop_threshold=0.008, tp_pct=0.50, sl_pct=0.25, horizon=15),
    RetrConfig("lb1_th0p8_tp75_sl50", lookback=1, drop_threshold=0.008, tp_pct=0.75, sl_pct=0.50, horizon=20),
    RetrConfig("lb1_th1p0_tp75_sl40", lookback=1, drop_threshold=0.010, tp_pct=0.75, sl_pct=0.40, horizon=20),
    # Raras
    RetrConfig("lb1_th1p5_tp50_sl25", lookback=1, drop_threshold=0.015, tp_pct=0.50, sl_pct=0.25, horizon=20),

    # === Lookback 3 (3-bar drop) ===
    # Frecuentes
    RetrConfig("lb3_th0p3_tp25_sl40", lookback=3, drop_threshold=0.003, tp_pct=0.25, sl_pct=0.40, horizon=10),
    RetrConfig("lb3_th0p5_tp30_sl40", lookback=3, drop_threshold=0.005, tp_pct=0.30, sl_pct=0.40, horizon=15),
    RetrConfig("lb3_th0p5_tp50_sl50", lookback=3, drop_threshold=0.005, tp_pct=0.50, sl_pct=0.50, horizon=15),
    # Intermedias
    RetrConfig("lb3_th0p8_tp50_sl40", lookback=3, drop_threshold=0.008, tp_pct=0.50, sl_pct=0.40, horizon=15),
    RetrConfig("lb3_th1p0_tp75_sl50", lookback=3, drop_threshold=0.010, tp_pct=0.75, sl_pct=0.50, horizon=20),
    # Raras
    RetrConfig("lb3_th1p5_tp100_sl60", lookback=3, drop_threshold=0.015, tp_pct=1.00, sl_pct=0.60, horizon=25),

    # === Lookback 5 (5-bar drop) ===
    # Frecuentes
    RetrConfig("lb5_th0p5_tp30_sl50", lookback=5, drop_threshold=0.005, tp_pct=0.30, sl_pct=0.50, horizon=15),
    # Intermedias
    RetrConfig("lb5_th0p8_tp50_sl50", lookback=5, drop_threshold=0.008, tp_pct=0.50, sl_pct=0.50, horizon=20),
    RetrConfig("lb5_th1p0_tp75_sl60", lookback=5, drop_threshold=0.010, tp_pct=0.75, sl_pct=0.60, horizon=25),
    # Raras
    RetrConfig("lb5_th1p5_tp100_sl40", lookback=5, drop_threshold=0.015, tp_pct=1.00, sl_pct=0.40, horizon=30),

    # === SNAPBACK GRID (ChatGPT's low TP/SL strategy) ===
    # Hypothesis: after big drops, price stabilizes + small bounce
    # Strategy: capture small rebound, cut quickly if wrong

    # LB=3, th=1.2% - low TP/SL, short horizon
    RetrConfig("lb3_th1p2_tp25_sl25", lookback=3, drop_threshold=0.012, tp_pct=0.25, sl_pct=0.25, horizon=15),
    RetrConfig("lb3_th1p2_tp30_sl25", lookback=3, drop_threshold=0.012, tp_pct=0.30, sl_pct=0.25, horizon=15),
    RetrConfig("lb3_th1p2_tp50_sl25", lookback=3, drop_threshold=0.012, tp_pct=0.50, sl_pct=0.25, horizon=30),
    RetrConfig("lb3_th1p2_tp30_sl40", lookback=3, drop_threshold=0.012, tp_pct=0.30, sl_pct=0.40, horizon=30),

    # LB=3, th=1.3% - with longer horizons
    RetrConfig("lb3_th1p3_tp25_sl25", lookback=3, drop_threshold=0.013, tp_pct=0.25, sl_pct=0.25, horizon=15),
    RetrConfig("lb3_th1p3_tp30_sl25", lookback=3, drop_threshold=0.013, tp_pct=0.30, sl_pct=0.25, horizon=15),
    RetrConfig("lb3_th1p3_tp50_sl25", lookback=3, drop_threshold=0.013, tp_pct=0.50, sl_pct=0.25, horizon=60),
    RetrConfig("lb3_th1p3_tp30_sl40", lookback=3, drop_threshold=0.013, tp_pct=0.30, sl_pct=0.40, horizon=60),

    # LB=5, th=1.2-1.5% - 5-bar drops with various horizons
    RetrConfig("lb5_th1p2_tp25_sl25", lookback=5, drop_threshold=0.012, tp_pct=0.25, sl_pct=0.25, horizon=30),
    RetrConfig("lb5_th1p2_tp30_sl25", lookback=5, drop_threshold=0.012, tp_pct=0.30, sl_pct=0.25, horizon=30),
    RetrConfig("lb5_th1p5_tp30_sl40", lookback=5, drop_threshold=0.015, tp_pct=0.30, sl_pct=0.40, horizon=60),
    RetrConfig("lb5_th1p5_tp50_sl40", lookback=5, drop_threshold=0.015, tp_pct=0.50, sl_pct=0.40, horizon=120),
]


# Extended grid with confirmation signals
@dataclass
class RetrConfigV2:
    """Config with confirmation entry (wait for green bar after drop)."""
    name: str
    lookback: int
    drop_threshold: float
    tp_pct: float
    sl_pct: float
    horizon: int
    confirm_bars: int = 0  # 0=immediate, 1=wait for green bar, 2=wait for 2 green

    @property
    def breakeven_rate(self) -> float:
        return self.sl_pct / (self.tp_pct + self.sl_pct)


# Confirmation-based entries
GRID_V2: List[RetrConfigV2] = [
    # Wait for reversal confirmation (1 green bar)
    RetrConfigV2("confirm1_aggressive", lookback=1, drop_threshold=0.005, tp_pct=0.30, sl_pct=0.50, horizon=10, confirm_bars=1),
    RetrConfigV2("confirm1_balanced", lookback=1, drop_threshold=0.005, tp_pct=0.50, sl_pct=0.50, horizon=15, confirm_bars=1),
    RetrConfigV2("confirm1_big", lookback=1, drop_threshold=0.010, tp_pct=0.40, sl_pct=0.50, horizon=15, confirm_bars=1),

    # Wait for 2 green bars (stronger confirmation)
    RetrConfigV2("confirm2_aggressive", lookback=1, drop_threshold=0.005, tp_pct=0.30, sl_pct=0.60, horizon=12, confirm_bars=2),
    RetrConfigV2("confirm2_balanced", lookback=1, drop_threshold=0.008, tp_pct=0.50, sl_pct=0.50, horizon=15, confirm_bars=2),

    # Multi-bar drop with confirmation
    RetrConfigV2("3bar_confirm1", lookback=3, drop_threshold=0.010, tp_pct=0.40, sl_pct=0.50, horizon=15, confirm_bars=1),
    RetrConfigV2("5bar_confirm1", lookback=5, drop_threshold=0.015, tp_pct=0.50, sl_pct=0.50, horizon=20, confirm_bars=1),
]


def detect_drops(df: pd.DataFrame, lookback: int, threshold: float) -> pd.Series:
    """
    Detect bars where price dropped by threshold over lookback period.

    For lookback=1: close[i] / close[i-1] - 1 < -threshold
    For lookback=N: close[i] / close[i-N] - 1 < -threshold

    Returns boolean Series marking drop events.
    """
    if lookback == 1:
        returns = df['close'].pct_change()
    else:
        returns = df['close'] / df['close'].shift(lookback) - 1

    return returns < -threshold


def evaluate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    drop_size: float,
    tp_pct: float,
    sl_pct: float,
    horizon: int,
    costs: float = 0.005
) -> Tuple[str, float, int]:
    """
    Evaluate a single trade with:
    - Entry at open[entry_idx]
    - TP = entry + drop_size * tp_pct
    - SL = entry - drop_size * sl_pct
    - Exit at horizon bars if neither hit

    Returns: (outcome: 'TP'|'SL'|'EXP', pnl_pct, bars_held)
    """
    if entry_idx >= len(df):
        return ('SKIP', 0.0, 0)

    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price + drop_size * tp_pct
    sl_price = entry_price - drop_size * sl_pct

    for i in range(entry_idx, min(entry_idx + horizon, len(df))):
        bar = df.iloc[i]

        # Check TP hit (using high)
        if bar['high'] >= tp_price:
            pnl = (tp_price / entry_price - 1) - costs
            return ('TP', pnl, i - entry_idx + 1)

        # Check SL hit (using low)
        if bar['low'] <= sl_price:
            pnl = (sl_price / entry_price - 1) - costs
            return ('SL', pnl, i - entry_idx + 1)

    # Expired - exit at close of last bar
    if entry_idx + horizon - 1 < len(df):
        exit_price = df.iloc[entry_idx + horizon - 1]['close']
        pnl = (exit_price / entry_price - 1) - costs
        return ('EXP', pnl, horizon)

    return ('SKIP', 0.0, 0)


def run_config(df: pd.DataFrame, config: RetrConfig, costs: float = 0.005) -> dict:
    """
    Run backtest for a single configuration.

    Returns dict with results.
    """
    # Detect drops
    drops = detect_drops(df, config.lookback, config.drop_threshold)

    results = {
        'name': config.name,
        'lookback': config.lookback,
        'drop_threshold': config.drop_threshold,
        'tp_pct': config.tp_pct,
        'sl_pct': config.sl_pct,
        'horizon': config.horizon,
        'breakeven_rate': config.breakeven_rate,
        'signals': 0,
        'tp_count': 0,
        'sl_count': 0,
        'exp_count': 0,
        'skip_count': 0,
        'total_pnl': 0.0,
        'trades': []
    }

    # Process each drop event
    drop_indices = df.index[drops].tolist()

    for idx in drop_indices:
        # Get numeric position
        pos = df.index.get_loc(idx)

        # Calculate drop size
        if config.lookback == 1:
            drop_size = df.iloc[pos - 1]['close'] - df.iloc[pos]['close']
        else:
            drop_size = df.iloc[pos - config.lookback]['close'] - df.iloc[pos]['close']

        if drop_size <= 0:
            continue

        # Entry at next bar's open
        entry_pos = pos + 1

        outcome, pnl, bars = evaluate_trade(
            df, entry_pos, drop_size,
            config.tp_pct, config.sl_pct, config.horizon, costs
        )

        results['signals'] += 1

        if outcome == 'TP':
            results['tp_count'] += 1
        elif outcome == 'SL':
            results['sl_count'] += 1
        elif outcome == 'EXP':
            results['exp_count'] += 1
        else:
            results['skip_count'] += 1
            continue

        results['total_pnl'] += pnl
        results['trades'].append({
            'entry_idx': entry_pos,
            'outcome': outcome,
            'pnl': pnl,
            'bars': bars
        })

    # Calculate metrics
    total_trades = results['tp_count'] + results['sl_count'] + results['exp_count']
    if total_trades > 0:
        results['win_rate'] = results['tp_count'] / total_trades
        results['avg_pnl'] = results['total_pnl'] / total_trades
        results['avg_bars'] = sum(t['bars'] for t in results['trades']) / total_trades
    else:
        results['win_rate'] = 0
        results['avg_pnl'] = 0
        results['avg_bars'] = 0

    # Calculate expire-specific metrics (ChatGPT checklist)
    expire_trades = [t for t in results['trades'] if t['outcome'] == 'EXP']
    if expire_trades:
        # Net PnL already includes costs
        results['avg_expire_net'] = sum(t['pnl'] for t in expire_trades) / len(expire_trades)
        # Gross = net + costs
        results['avg_expire_gross'] = results['avg_expire_net'] + costs
    else:
        results['avg_expire_net'] = 0
        results['avg_expire_gross'] = 0

    return results


def run_grid(df: pd.DataFrame, configs: List[RetrConfig] = GRID, costs: float = 0.005) -> pd.DataFrame:
    """
    Run all configurations and return results DataFrame.
    Includes p_win, p_loss, p_expire and EV per trade as ChatGPT suggested.
    """
    all_results = []

    for config in configs:
        result = run_config(df, config, costs)
        total = result['tp_count'] + result['sl_count'] + result['exp_count']

        # Calculate probabilities
        p_win = result['tp_count'] / total if total > 0 else 0
        p_loss = result['sl_count'] / total if total > 0 else 0
        p_expire = result['exp_count'] / total if total > 0 else 0

        # EV per trade (average P&L)
        ev = result['avg_pnl'] if total > 0 else 0

        # Determine family based on config name and threshold
        name = result['name']
        th = result['drop_threshold']
        tp = result['tp_pct']

        # Snapback configs have low TP (25-50%) and high threshold (1.2%+)
        if th >= 0.012 and tp <= 0.50:
            family = "snap"  # Snapback strategy
        elif th <= 0.005:
            family = "freq"
        elif th <= 0.010:
            family = "inter"
        elif th <= 0.013:
            family = "bridge"
        else:
            family = "rare"

        all_results.append({
            'name': result['name'],
            'family': family,
            'lb': result['lookback'],
            'th': f"{result['drop_threshold']*100:.1f}%",
            'tp': f"{result['tp_pct']*100:.0f}%",
            'sl': f"{result['sl_pct']*100:.0f}%",
            'h': result['horizon'],
            'BE': f"{result['breakeven_rate']*100:.0f}%",
            'n': total,
            'p_win': f"{p_win*100:.0f}%" if total > 0 else '-',
            'p_loss': f"{p_loss*100:.0f}%" if total > 0 else '-',
            'p_exp': f"{p_expire*100:.0f}%" if total > 0 else '-',
            'EV': f"{ev*100:.2f}%" if total > 0 else '-',
            'exp_g': f"{result['avg_expire_gross']*100:.2f}%" if result['exp_count'] > 0 else '-',
            'exp_n': f"{result['avg_expire_net']*100:.2f}%" if result['exp_count'] > 0 else '-',
            'total': f"{result['total_pnl']*100:.2f}%",
        })

    return pd.DataFrame(all_results)


def find_confirmation_entry(df: pd.DataFrame, drop_pos: int, confirm_bars: int, max_wait: int = 5) -> Optional[int]:
    """
    Find entry position after confirmation (green bars).

    Returns position where we should enter, or None if confirmation not found within max_wait.
    """
    if confirm_bars == 0:
        return drop_pos + 1  # Immediate entry

    green_count = 0
    for i in range(drop_pos + 1, min(drop_pos + 1 + max_wait, len(df))):
        bar = df.iloc[i]
        if bar['close'] > bar['open']:  # Green bar
            green_count += 1
            if green_count >= confirm_bars:
                return i + 1  # Enter at next bar's open
        else:
            green_count = 0  # Reset if red bar

    return None  # No confirmation within wait period


def run_config_v2(df: pd.DataFrame, config: RetrConfigV2, costs: float = 0.005) -> dict:
    """
    Run backtest for a configuration with confirmation entry.
    """
    drops = detect_drops(df, config.lookback, config.drop_threshold)

    results = {
        'name': config.name,
        'lookback': config.lookback,
        'drop_threshold': config.drop_threshold,
        'tp_pct': config.tp_pct,
        'sl_pct': config.sl_pct,
        'horizon': config.horizon,
        'confirm_bars': config.confirm_bars,
        'breakeven_rate': config.breakeven_rate,
        'signals': 0,
        'confirmed': 0,
        'tp_count': 0,
        'sl_count': 0,
        'exp_count': 0,
        'skip_count': 0,
        'total_pnl': 0.0,
        'trades': []
    }

    drop_indices = df.index[drops].tolist()

    for idx in drop_indices:
        pos = df.index.get_loc(idx)
        results['signals'] += 1

        # Calculate drop size
        if config.lookback == 1:
            drop_size = df.iloc[pos - 1]['close'] - df.iloc[pos]['close']
        else:
            drop_size = df.iloc[pos - config.lookback]['close'] - df.iloc[pos]['close']

        if drop_size <= 0:
            continue

        # Find confirmation entry
        entry_pos = find_confirmation_entry(df, pos, config.confirm_bars)
        if entry_pos is None:
            results['skip_count'] += 1
            continue

        results['confirmed'] += 1

        outcome, pnl, bars = evaluate_trade(
            df, entry_pos, drop_size,
            config.tp_pct, config.sl_pct, config.horizon, costs
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
            'entry_idx': entry_pos,
            'outcome': outcome,
            'pnl': pnl,
            'bars': bars
        })

    total_trades = results['tp_count'] + results['sl_count'] + results['exp_count']
    if total_trades > 0:
        results['win_rate'] = results['tp_count'] / total_trades
        results['avg_pnl'] = results['total_pnl'] / total_trades
        results['avg_bars'] = sum(t['bars'] for t in results['trades']) / total_trades
    else:
        results['win_rate'] = 0
        results['avg_pnl'] = 0
        results['avg_bars'] = 0

    return results


def run_grid_v2(df: pd.DataFrame, configs: List[RetrConfigV2] = GRID_V2, costs: float = 0.005) -> pd.DataFrame:
    """Run grid with confirmation entries."""
    all_results = []

    for config in configs:
        result = run_config_v2(df, config, costs)
        all_results.append({
            'name': result['name'],
            'confirm': result['confirm_bars'],
            'drop': f"{result['drop_threshold']*100:.1f}%",
            'tp': f"{result['tp_pct']*100:.0f}%",
            'sl': f"{result['sl_pct']*100:.0f}%",
            'h': result['horizon'],
            'be': f"{result['breakeven_rate']*100:.1f}%",
            'signals': result['signals'],
            'confirmed': result['confirmed'],
            'trades': result['tp_count'] + result['sl_count'] + result['exp_count'],
            'wins': result['tp_count'],
            'losses': result['sl_count'],
            'exp': result['exp_count'],
            'win_rate': f"{result['win_rate']*100:.1f}%" if result['signals'] > 0 else '-',
            'avg_pnl': f"{result['avg_pnl']*100:.3f}%" if result['signals'] > 0 else '-',
            'total_pnl': f"{result['total_pnl']*100:.3f}%",
        })

    return pd.DataFrame(all_results)


def load_trades(pair: str = "BTC/USD") -> Optional[pd.DataFrame]:
    """Load raw trade data from parquet files."""
    data_dir = Path("data/crypto/trade")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return None

    # Find all trade parquet files
    files = sorted(data_dir.rglob("*.parquet"))

    if not files:
        print(f"No trade data files found")
        return None

    print(f"Found {len(files)} trade files")

    # Load and concatenate all files
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Filter for the requested pair
            df = df[df['pair'] == pair]
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        print(f"No trades found for {pair}")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} trades for {pair}")

    return df


def trades_to_ohlc(trades: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    """
    Aggregate trades into OHLC bars.

    trades: DataFrame with columns [ts, pair, side, price, qty, trade_id]
    interval: pandas resample interval (e.g., '1min', '5min')
    """
    # Convert timestamp (milliseconds) to datetime
    trades['timestamp'] = pd.to_datetime(trades['ts'], unit='ms')
    trades = trades.set_index('timestamp')
    trades = trades.sort_index()

    # Resample to OHLC
    ohlc = trades['price'].resample(interval).ohlc()
    volume = trades['qty'].resample(interval).sum()

    # Combine
    df = ohlc.copy()
    df['volume'] = volume

    # Forward fill missing bars (market was quiet)
    df = df.ffill()

    # Drop any remaining NaN
    df = df.dropna()

    return df


def load_data(pair: str = "BTC/USD", interval: str = "1min") -> Optional[pd.DataFrame]:
    """Load trade data and convert to OHLC bars."""
    trades = load_trades(pair)

    if trades is None or len(trades) == 0:
        return None

    print(f"Aggregating trades to {interval} OHLC bars...")
    ohlc = trades_to_ohlc(trades, interval)
    print(f"Created {len(ohlc)} bars")

    return ohlc


def main():
    """Run grid analysis and display results."""
    print("=" * 80)
    print("RETRACEMENT STRATEGY GRID SEARCH")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    df = load_data("BTC/USD")

    if df is None or len(df) == 0:
        print("No data available. Run the recorder to collect more data.")
        print()
        print("Grid configurations defined:")
        print("-" * 60)
        for i, config in enumerate(GRID):
            print(f"{i+1:2}. {config.name:20} | lb={config.lookback} drop={config.drop_threshold*100:.1f}% "
                  f"tp={config.tp_pct*100:.0f}% sl={config.sl_pct*100:.0f}% h={config.horizon} "
                  f"BE={config.breakeven_rate*100:.1f}%")
        return

    print(f"Loaded {len(df)} bars")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print()

    # Run grid
    print("Running ChatGPT's 18-config grid (0.5% costs)...")
    print()

    results = run_grid(df, GRID, costs=0.005)

    # Convert total to float for sorting
    results['pnl_sort'] = results['total'].str.rstrip('%').astype(float)

    # Display by family
    families = [
        ('freq', 'FRECUENTES (0.3-0.5%) - high cost sensitivity'),
        ('inter', 'INTERMEDIAS (0.8-1.0%) - core range'),
        ('snap', 'SNAPBACK (1.2%+, low TP) - ChatGPT strategy ★'),
        ('bridge', 'BRIDGE (1.2-1.3%, high TP) - original'),
        ('rare', 'RARAS (1.5%+, high TP) - edge appears here first'),
    ]
    for family, label in families:
        family_df = results[results['family'] == family].sort_values('pnl_sort', ascending=False)
        if len(family_df) > 0:
            print(f"\n{label}:")
            print("-" * 120)
            print(family_df.drop(columns=['pnl_sort', 'family']).to_string(index=False))

    print()

    # Run confirmation grid
    print("=" * 80)
    print("Running CONFIRMATION ENTRY grid (wait for green bars)...")
    print()

    results_v2 = run_grid_v2(df, GRID_V2, costs=0.005)

    print("CONFIRMATION ENTRY RESULTS:")
    print("-" * 100)

    results_v2['pnl_sort'] = results_v2['total_pnl'].str.rstrip('%').astype(float)
    results_v2 = results_v2.sort_values('pnl_sort', ascending=False)

    print(results_v2.drop(columns=['pnl_sort']).to_string(index=False))
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find configs with positive expectancy
    positive = results[results['pnl_sort'] > 0]
    positive_v2 = results_v2[results_v2['pnl_sort'] > 0]

    if len(positive) > 0 or len(positive_v2) > 0:
        print(f"\n✅ Configs with positive P&L:")
        for _, row in positive.iterrows():
            print(f"  - {row['name']}: {row['total']} ({row['n']} trades, EV={row['EV']})")
        for _, row in positive_v2.iterrows():
            print(f"  - {row['name']}: {row['total_pnl']} ({row['trades']} trades)")
    else:
        print("\n❌ NO CONFIGS WITH POSITIVE P&L")
        print("   The retracement pattern does not appear to have edge in this data.")

    # Best performing (least bad)
    all_pnl = list(zip(results['name'], results['pnl_sort'], results['n'])) + \
              list(zip(results_v2['name'], results_v2['pnl_sort'], results_v2['trades']))
    all_pnl.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 configs (least bad):")
    for name, pnl, trades in all_pnl[:5]:
        print(f"  {name}: {pnl:.2f}% ({trades} trades)")

    # Configs with most trades (for statistical significance)
    high_n = results[results['n'] >= 5]
    if len(high_n) > 0:
        print(f"\nConfigs with 5+ trades:")
        for _, row in high_n.iterrows():
            print(f"  - {row['name']}: n={row['n']}, p_win={row['p_win']}, p_loss={row['p_loss']}, EV={row['EV']}")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    total_trades_immediate = results['n'].sum()
    total_trades_confirm = results_v2['trades'].sum()
    total_pnl_immediate = results['pnl_sort'].sum()
    total_pnl_confirm = results_v2['pnl_sort'].sum()

    print(f"\nImmediate entry: {total_trades_immediate} trades, {total_pnl_immediate:.2f}% cumulative P&L")
    print(f"Confirmation entry: {total_trades_confirm} trades, {total_pnl_confirm:.2f}% cumulative P&L")

    if total_pnl_immediate < 0 and total_pnl_confirm < 0:
        print("\n⚠️  Both strategies losing money across all configurations.")
        print("⚠️  Consider abandoning retracement or testing longer timeframes.")

    # Quick filter advice
    print("\n" + "=" * 80)
    print("QUICK FILTER (ChatGPT's advice)")
    print("=" * 80)
    print("1. Discard configs where p_win << BE (breakeven)")
    print("2. Focus on 'rare' family first - edge appears there if it exists")
    print("3. Need 30+ trades per config for statistical significance")
    print("4. Current data: ~30 hours - need 1-2 weeks for reliable results")


if __name__ == "__main__":
    main()
