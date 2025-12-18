#!/usr/bin/env python3
"""
ML Analysis Script for Kraken Scalper
=====================================

Loads parquet market data, creates features, trains regression models,
and evaluates trading strategies.

Usage:
    python scripts/ml_analysis.py --data-dir data --pair BTC/USD --target 5m --plot

Author: Generated with Claude Code
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed, skipping XGBoost model")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed, skipping LightGBM model")

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("Warning: ta library not installed, using manual indicator calculations")


# =============================================================================
# Data Loading
# =============================================================================

def load_parquet_files(
    data_dir: Path,
    category: str,
    data_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all parquet files for a given category and data type.

    Args:
        data_dir: Base data directory
        category: 'crypto' or 'stocks'
        data_type: 'ticker', 'ohlc', 'trade', or 'book'
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with all data concatenated
    """
    path = data_dir / category / data_type

    if not path.exists():
        print(f"Warning: Path {path} does not exist")
        return pd.DataFrame()

    files = sorted(path.glob("**/*.parquet"))

    if not files:
        print(f"Warning: No parquet files found in {path}")
        return pd.DataFrame()

    # Filter by date if specified
    if start_date or end_date:
        filtered_files = []
        for f in files:
            # Extract date from path: data/crypto/ticker/2025-12-17/13.parquet
            date_str = f.parent.name
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if start_date and file_date < datetime.strptime(start_date, "%Y-%m-%d"):
                    continue
                if end_date and file_date > datetime.strptime(end_date, "%Y-%m-%d"):
                    continue
                filtered_files.append(f)
            except ValueError:
                filtered_files.append(f)  # Keep files with non-date directories
        files = filtered_files

    dfs = []
    for f in tqdm(files, desc=f"Loading {category}/{data_type}", leave=False):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_ticker_data(data_dir: Path, pair: str, category: str = "crypto") -> pd.DataFrame:
    """Load and filter ticker data for a specific pair."""
    df = load_parquet_files(data_dir, category, "ticker")

    if df.empty:
        return df

    # Filter by pair
    df = df[df['pair'] == pair].copy()

    if df.empty:
        print(f"Warning: No data found for pair {pair}")
        return df

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates(subset=['ts'], keep='last')

    return df


def load_ohlc_data(data_dir: Path, pair: str, category: str = "crypto") -> pd.DataFrame:
    """Load and filter OHLC data for a specific pair."""
    df = load_parquet_files(data_dir, category, "ohlc")

    if df.empty:
        return df

    df = df[df['pair'] == pair].copy()

    if df.empty:
        return df

    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['ts'], keep='last')

    return df


def load_book_data(data_dir: Path, pair: str, category: str = "crypto") -> pd.DataFrame:
    """Load and parse order book data for a specific pair."""
    df = load_parquet_files(data_dir, category, "book")

    if df.empty:
        return df

    df = df[df['pair'] == pair].copy()

    if df.empty:
        return df

    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)

    # Parse JSON book data
    def parse_book_side(json_str: str) -> List[Tuple[float, float]]:
        try:
            return json.loads(json_str) if pd.notna(json_str) else []
        except:
            return []

    df['bids'] = df['bids_json'].apply(parse_book_side)
    df['asks'] = df['asks_json'].apply(parse_book_side)

    return df


# =============================================================================
# Feature Engineering
# =============================================================================

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, Signal, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to dataframe."""
    df = df.copy()

    # Price column (use 'last' from ticker or 'close' from ohlc)
    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col]

    # RSI
    df['rsi'] = compute_rsi(prices, 14)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(prices)
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) &
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

    # Bollinger Bands
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = compute_bollinger_bands(prices)
    df['bb_position'] = (prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # Moving Averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = prices.rolling(window=period).mean()
        df[f'ema_{period}'] = prices.ewm(span=period, adjust=False).mean()
        df[f'price_vs_sma_{period}'] = prices / df[f'sma_{period}'] - 1

    # ATR (if OHLC data available)
    if all(col in df.columns for col in ['high', 'low', 'close']):
        df['atr'] = compute_atr(df['high'], df['low'], df['close'], 14)
        df['atr_pct'] = df['atr'] / df['close']

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and return features."""
    df = df.copy()

    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col]

    # Returns at different horizons
    for period in [1, 2, 3, 5, 10, 15, 30]:
        df[f'return_{period}'] = prices.pct_change(period)
        df[f'momentum_{period}'] = prices / prices.shift(period) - 1

    # Volatility
    df['volatility_5'] = prices.pct_change().rolling(5).std()
    df['volatility_10'] = prices.pct_change().rolling(10).std()
    df['volatility_20'] = prices.pct_change().rolling(20).std()

    # Rate of change
    df['roc_5'] = (prices - prices.shift(5)) / prices.shift(5) * 100
    df['roc_10'] = (prices - prices.shift(10)) / prices.shift(10) * 100

    # Price acceleration
    df['acceleration'] = df['return_1'] - df['return_1'].shift(1)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    df = df.copy()

    if 'volume' not in df.columns:
        return df

    volume = df['volume']

    # Volume moving averages
    df['volume_ma_5'] = volume.rolling(5).mean()
    df['volume_ma_20'] = volume.rolling(20).mean()

    # Volume ratio
    df['volume_ratio'] = volume / df['volume_ma_20']
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)

    # Volume trend
    df['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()

    # VWAP deviation (if vwap available)
    if 'vwap' in df.columns:
        price_col = 'close' if 'close' in df.columns else 'last'
        df['vwap_deviation'] = df[price_col] / df['vwap'] - 1

    return df


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bid-ask spread features."""
    df = df.copy()

    if 'bid' not in df.columns or 'ask' not in df.columns:
        return df

    df['spread'] = df['ask'] - df['bid']
    df['spread_pct'] = df['spread'] / ((df['ask'] + df['bid']) / 2) * 100
    df['mid_price'] = (df['ask'] + df['bid']) / 2

    # Spread statistics
    df['spread_ma'] = df['spread_pct'].rolling(20).mean()
    df['spread_std'] = df['spread_pct'].rolling(20).std()
    df['spread_zscore'] = (df['spread_pct'] - df['spread_ma']) / df['spread_std']

    return df


def add_book_features(ticker_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    """Add order book features by merging with ticker data."""
    if book_df.empty:
        return ticker_df

    df = ticker_df.copy()

    # Calculate book metrics
    def calc_book_metrics(row) -> dict:
        bids = row.get('bids', [])
        asks = row.get('asks', [])

        if not bids or not asks:
            return {'bid_depth': np.nan, 'ask_depth': np.nan, 'imbalance': np.nan}

        bid_depth = sum(b[1] for b in bids[:10]) if bids else 0
        ask_depth = sum(a[1] for a in asks[:10]) if asks else 0
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': imbalance
        }

    # Merge on nearest timestamp
    book_df = book_df.sort_values('ts')
    df = df.sort_values('ts')

    # Use merge_asof for nearest timestamp matching
    df = pd.merge_asof(
        df,
        book_df[['ts', 'bids', 'asks']],
        on='ts',
        direction='nearest',
        tolerance=60000  # 60 second tolerance
    )

    # Calculate metrics
    metrics = df.apply(calc_book_metrics, axis=1, result_type='expand')
    df = pd.concat([df, metrics], axis=1)

    # Clean up
    df = df.drop(columns=['bids', 'asks'], errors='ignore')

    return df


def create_features(
    ticker_df: pd.DataFrame,
    ohlc_df: Optional[pd.DataFrame] = None,
    book_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create all features from raw data."""

    # Start with ticker data
    df = ticker_df.copy()

    # Add OHLC features if available
    if ohlc_df is not None and not ohlc_df.empty:
        # Merge OHLC on nearest timestamp
        ohlc_cols = ['ts', 'open', 'high', 'low', 'close', 'trades']
        ohlc_subset = ohlc_df[ohlc_cols].copy()
        df = pd.merge_asof(
            df.sort_values('ts'),
            ohlc_subset.sort_values('ts'),
            on='ts',
            direction='nearest',
            tolerance=60000
        )

    # Add order book features
    if book_df is not None and not book_df.empty:
        df = add_book_features(df, book_df)

    # Add technical indicators
    df = add_technical_features(df)

    # Add momentum features
    df = add_momentum_features(df)

    # Add volume features
    df = add_volume_features(df)

    # Add spread features
    df = add_spread_features(df)

    return df


# =============================================================================
# Target Creation
# =============================================================================

def create_targets(df: pd.DataFrame, horizons: List[int] = [5, 15, 30, 60]) -> pd.DataFrame:
    """Create forward return targets for different horizons."""
    df = df.copy()

    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col]

    for h in horizons:
        # Forward return
        df[f'target_{h}m'] = prices.shift(-h) / prices - 1

        # Direction (1 = up, 0 = down)
        df[f'direction_{h}m'] = (df[f'target_{h}m'] > 0).astype(int)

    return df


# =============================================================================
# Model Training
# =============================================================================

@dataclass
class ModelResult:
    name: str
    model: object
    r2_scores: List[float]
    mae_scores: List[float]
    directional_accuracy: List[float]
    feature_importance: Dict[str, float]


def train_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_splits: int = 5,
) -> List[ModelResult]:
    """Train multiple models using TimeSeriesSplit cross-validation."""

    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values

    # Remove rows with NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(X) < 100:
        print(f"Warning: Only {len(X)} samples available for training")
        return []

    print(f"\nTraining with {len(X)} samples, {len(feature_cols)} features")

    # Models to train
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=10,
            random_state=42,
            verbosity=0
        )

    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=10,
            random_state=42,
            verbose=-1
        )

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        r2_scores = []
        mae_scores = []
        dir_acc = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            dir_accuracy = np.mean((y_pred > 0) == (y_test > 0))

            r2_scores.append(r2)
            mae_scores.append(mae)
            dir_acc.append(dir_accuracy)

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_cols, model.feature_importances_))
        else:
            importance = {}

        result = ModelResult(
            name=name,
            model=model,
            r2_scores=r2_scores,
            mae_scores=mae_scores,
            directional_accuracy=dir_acc,
            feature_importance=importance
        )
        results.append(result)

        print(f"  R² = {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
        print(f"  MAE = {np.mean(mae_scores):.6f} (+/- {np.std(mae_scores):.6f})")
        print(f"  Directional Accuracy = {np.mean(dir_acc):.2%} (+/- {np.std(dir_acc):.2%})")

    return results


# =============================================================================
# Backtesting
# =============================================================================

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def backtest_strategy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    price_col: str = 'last',
    threshold: float = 0.0001,  # Min predicted return to trade
    transaction_cost: float = 0.001,  # 0.1% per trade
) -> BacktestResult:
    """
    Simple backtest: go long when predicted return > threshold,
    flat otherwise.
    """
    prices = df[price_col].values
    actual_returns = np.diff(prices) / prices[:-1]

    # Align predictions with returns
    predictions = predictions[:-1]

    # Trading signals: 1 = long, 0 = flat
    signals = (predictions > threshold).astype(float)

    # Strategy returns
    strategy_returns = signals * actual_returns

    # Apply transaction costs on position changes
    position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
    costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - costs

    # Equity curve
    equity = np.cumprod(1 + strategy_returns)
    equity_series = pd.Series(equity, index=df.index[1:len(equity)+1])

    # Metrics
    total_return = equity[-1] - 1 if len(equity) > 0 else 0

    # Sharpe ratio (annualized, assuming 1-min data)
    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(525600)  # mins per year
    else:
        sharpe = 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    # Win rate
    winning_trades = np.sum((signals == 1) & (actual_returns > 0))
    total_trades = np.sum(signals == 1)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return BacktestResult(
        equity_curve=equity_series,
        returns=pd.Series(strategy_returns),
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=int(total_trades)
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_feature_importance(results: List[ModelResult], top_n: int = 15, save_path: Optional[Path] = None):
    """Plot feature importance for all models."""
    n_models = len([r for r in results if r.feature_importance])
    if n_models == 0:
        print("No feature importance data available")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        if not result.feature_importance:
            continue

        # Sort by importance
        sorted_features = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'{result.name} - Top {top_n} Features')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")

    plt.show()


def plot_backtest_results(
    backtest: BacktestResult,
    pair: str,
    save_path: Optional[Path] = None
):
    """Plot backtest equity curve and metrics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    ax1 = axes[0]
    backtest.equity_curve.plot(ax=ax1, label='Strategy')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f'{pair} - Strategy Equity Curve')
    ax1.set_ylabel('Equity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Returns distribution
    ax2 = axes[1]
    backtest.returns.hist(bins=50, ax=ax2, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_title('Strategy Returns Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')

    # Add metrics text
    metrics_text = (
        f"Total Return: {backtest.total_return:.2%}\n"
        f"Sharpe Ratio: {backtest.sharpe_ratio:.2f}\n"
        f"Max Drawdown: {backtest.max_drawdown:.2%}\n"
        f"Win Rate: {backtest.win_rate:.2%}\n"
        f"Trades: {backtest.n_trades}"
    )
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved backtest plot to {save_path}")

    plt.show()


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
):
    """Scatter plot of predictions vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.set_title(f'{model_name} - Predictions vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# =============================================================================
# Main
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude targets and metadata)."""
    exclude_patterns = [
        'target_', 'direction_', 'datetime', 'ts', 'pair',
        'bids', 'asks', '_json'
    ]

    feature_cols = []
    for col in df.columns:
        if not any(pattern in col for pattern in exclude_patterns):
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                feature_cols.append(col)

    return feature_cols


def main():
    parser = argparse.ArgumentParser(description='ML Analysis for Trading Data')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--pair', type=str, default='BTC/USD', help='Trading pair')
    parser.add_argument('--category', type=str, default='crypto', choices=['crypto', 'stocks'])
    parser.add_argument('--target', type=str, default='5m', help='Target horizon (5m, 15m, 30m, 1h)')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save-model', type=str, help='Save best model to path')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory for plots')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Parse target horizon
    target_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60}
    if args.target not in target_map:
        print(f"Invalid target: {args.target}. Use one of: {list(target_map.keys())}")
        return
    target_horizon = target_map[args.target]
    target_col = f'target_{target_horizon}m'

    print(f"\n{'='*60}")
    print(f"ML Analysis for {args.pair}")
    print(f"Target: {args.target} forward return")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    ticker_df = load_ticker_data(data_dir, args.pair, args.category)

    if ticker_df.empty:
        print(f"No ticker data found for {args.pair}")
        # List available pairs
        all_ticker = load_parquet_files(data_dir, args.category, "ticker")
        if not all_ticker.empty:
            print(f"Available pairs: {sorted(all_ticker['pair'].unique())}")
        return

    print(f"Loaded {len(ticker_df)} ticker records")

    # Load supplementary data
    ohlc_df = load_ohlc_data(data_dir, args.pair, args.category)
    print(f"Loaded {len(ohlc_df)} OHLC records")

    book_df = load_book_data(data_dir, args.pair, args.category)
    print(f"Loaded {len(book_df)} order book records")

    # Create features
    print("\nCreating features...")
    df = create_features(ticker_df, ohlc_df, book_df)

    # Create targets
    print("Creating targets...")
    df = create_targets(df, horizons=[5, 15, 30, 60])

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Generated {len(feature_cols)} features")

    # Remove rows with NaN targets
    df = df.dropna(subset=[target_col])
    print(f"Training samples after removing NaN: {len(df)}")

    if len(df) < 100:
        print("Not enough data for training. Need at least 100 samples.")
        return

    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    results = train_models(df, target_col, feature_cols, n_splits=5)

    if not results:
        print("No models trained successfully")
        return

    # Find best model
    best_result = max(results, key=lambda r: np.mean(r.directional_accuracy))
    print(f"\nBest model: {best_result.name}")

    # Print top features
    if best_result.feature_importance:
        print(f"\nTop 10 Features ({best_result.name}):")
        sorted_features = sorted(
            best_result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for i, (feat, imp) in enumerate(sorted_features, 1):
            print(f"  {i:2}. {feat:30} {imp:.4f}")

    # Backtest
    print("\n" + "="*60)
    print("BACKTESTING")
    print("="*60)

    # Get predictions on full dataset for backtest
    X = df[feature_cols].values
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    df_clean = df[mask].copy()

    predictions = best_result.model.predict(X_clean)

    backtest = backtest_strategy(
        df_clean,
        predictions,
        price_col='last',
        threshold=0.0001,
        transaction_cost=0.001
    )

    print(f"\nBacktest Results ({best_result.name}):")
    print(f"  Total Return: {backtest.total_return:.2%}")
    print(f"  Sharpe Ratio: {backtest.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {backtest.max_drawdown:.2%}")
    print(f"  Win Rate: {backtest.win_rate:.2%}")
    print(f"  Number of Trades: {backtest.n_trades}")

    # Save model
    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump({
            'model': best_result.model,
            'feature_cols': feature_cols,
            'pair': args.pair,
            'target': args.target
        }, model_path)
        print(f"\nModel saved to {model_path}")

    # Plots
    if args.plot:
        print("\nGenerating plots...")

        # Feature importance
        plot_feature_importance(
            results,
            top_n=15,
            save_path=output_dir / f"{args.pair.replace('/', '_')}_feature_importance.png"
        )

        # Backtest results
        plot_backtest_results(
            backtest,
            args.pair,
            save_path=output_dir / f"{args.pair.replace('/', '_')}_backtest.png"
        )

        # Predictions vs actual
        y_actual = df_clean[target_col].values
        plot_predictions_vs_actual(
            y_actual,
            predictions,
            best_result.name,
            save_path=output_dir / f"{args.pair.replace('/', '_')}_predictions.png"
        )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
