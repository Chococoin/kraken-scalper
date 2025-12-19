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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
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


def load_kraken_ohlc_data(data_dir: Path, pair: str, interval: int = 1) -> pd.DataFrame:
    """
    Load OHLC data from Kraken REST API parquet files.

    These files are stored at: data/kraken/{pair_safe}/{interval}.parquet
    Example: data/kraken/BTC_USD/1.parquet

    Args:
        data_dir: Base data directory
        pair: Trading pair (e.g., 'BTC/USD')
        interval: Candle interval in minutes (default: 1)

    Returns:
        DataFrame with OHLC data ready for ML analysis
    """
    pair_safe = pair.replace("/", "_")
    file_path = data_dir / "kraken" / pair_safe / f"{interval}.parquet"

    if not file_path.exists():
        print(f"Warning: Kraken OHLC file not found: {file_path}")
        # List available pairs
        kraken_dir = data_dir / "kraken"
        if kraken_dir.exists():
            available = [d.name.replace("_", "/") for d in kraken_dir.iterdir() if d.is_dir()]
            print(f"Available pairs: {sorted(available)}")
        return pd.DataFrame()

    df = pd.read_parquet(file_path)

    if df.empty:
        return df

    # Convert timestamp from ms to datetime
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['ts'], keep='last')

    # Rename columns to match expected format
    # Kraken REST has: ts, pair, open, high, low, close, volume, vwap, trades
    # We need: datetime, open, high, low, close, volume, and optionally last (price)
    df['last'] = df['close']  # Use close as current price

    print(f"Loaded {len(df)} candles from {file_path}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def list_kraken_pairs(data_dir: Path) -> List[str]:
    """List available pairs in Kraken REST data."""
    kraken_dir = data_dir / "kraken"
    if not kraken_dir.exists():
        return []
    return sorted([d.name.replace("_", "/") for d in kraken_dir.iterdir() if d.is_dir()])


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


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Compute Stochastic Oscillator (%K and %D).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period for high/low range
        smooth_k: Smoothing period for %K
        smooth_d: Smoothing period for %D (signal line)

    Returns:
        Tuple of (%K smoothed, %D signal line)
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    # Raw %K
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Smoothed %K
    k_smooth = k.rolling(window=smooth_k).mean()

    # %D (signal line)
    d = k_smooth.rolling(window=smooth_d).mean()

    return k_smooth, d


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14
) -> pd.Series:
    """Compute Money Flow Index (volume-weighted RSI).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        period: Lookback period

    Returns:
        MFI values (0-100 scale)
    """
    # Typical price
    typical_price = (high + low + close) / 3

    # Money flow
    money_flow = typical_price * volume

    # Direction
    delta = typical_price.diff()

    # Positive and negative money flow
    positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
    negative_flow = money_flow.where(delta < 0, 0).abs().rolling(window=period).sum()

    # Money flow ratio and index
    mfr = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + mfr))

    return mfi


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Average Directional Index (ADX).

    ADX measures trend STRENGTH (not direction):
    - ADX < 20: Weak/no trend (good for mean reversion)
    - ADX 20-40: Developing trend
    - ADX > 40: Strong trend (good for trend following)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default 14)

    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    atr = tr.rolling(window=period).mean()
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()

    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    OBV is a cumulative indicator that adds volume on up days
    and subtracts volume on down days. It shows accumulation/distribution.

    - Rising OBV with rising price = uptrend confirmed
    - Falling OBV with rising price = potential reversal (distribution)
    - Rising OBV with falling price = potential reversal (accumulation)

    Args:
        close: Close prices
        volume: Trading volume

    Returns:
        OBV values (cumulative)
    """
    # Direction: +1 if price up, -1 if price down, 0 if unchanged
    direction = np.sign(close.diff())

    # OBV = cumulative sum of signed volume
    obv = (direction * volume).cumsum()

    return obv


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


def add_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add oscillator and trend strength indicators.

    Indicators added:
    - Stochastic: Uses high/low range (RSI only uses close)
    - MFI: Incorporates volume (volume-weighted RSI)
    - ADX: Trend strength (not direction) - good for filtering
    - OBV: On-Balance Volume for accumulation/distribution
    """
    df = df.copy()

    price_col = 'close' if 'close' in df.columns else 'last'

    # Need OHLC data for these indicators
    if not all(col in df.columns for col in ['high', 'low']):
        return df

    high = df['high']
    low = df['low']
    close = df[price_col]

    # Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = compute_stochastic(high, low, close)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    df['stoch_cross_up'] = ((df['stoch_k'] > df['stoch_d']) &
                            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
    df['stoch_cross_down'] = ((df['stoch_k'] < df['stoch_d']) &
                              (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))).astype(int)

    # ADX - Average Directional Index (trend strength)
    df['adx'], df['plus_di'], df['minus_di'] = compute_adx(high, low, close)
    df['adx_weak_trend'] = (df['adx'] < 20).astype(int)  # Good for mean reversion
    df['adx_strong_trend'] = (df['adx'] > 40).astype(int)  # Good for trend following
    df['di_cross_up'] = ((df['plus_di'] > df['minus_di']) &
                         (df['plus_di'].shift(1) <= df['minus_di'].shift(1))).astype(int)
    df['di_cross_down'] = ((df['plus_di'] < df['minus_di']) &
                           (df['plus_di'].shift(1) >= df['minus_di'].shift(1))).astype(int)

    # MFI - Money Flow Index (only if volume available)
    if 'volume' in df.columns:
        df['mfi'] = compute_mfi(high, low, close, df['volume'])
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)

        # OBV - On-Balance Volume
        df['obv'] = compute_obv(close, df['volume'])
        # OBV trend (is OBV rising or falling?)
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_sma']).astype(int)  # 1 = accumulation
        # OBV divergence signal (price down but OBV up = bullish divergence)
        price_down = (close < close.shift(5)).astype(int)
        obv_up = (df['obv'] > df['obv'].shift(5)).astype(int)
        df['obv_bull_divergence'] = (price_down & obv_up).astype(int)

    return df


# =============================================================================
# Multi-Timeframe Indicators
# =============================================================================

def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLC data to a higher timeframe.

    Args:
        df: DataFrame with OHLC data (must have 'ts' or datetime index)
        timeframe: Target timeframe ('5T', '15T', '1H')

    Returns:
        Resampled OHLC DataFrame
    """
    df = df.copy()

    # Ensure datetime index
    if 'ts' in df.columns:
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('datetime')
    elif not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()  # Can't resample without datetime

    # Only keep OHLCV columns for resampling
    ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in ohlc_cols if c in df.columns]

    if not available_cols:
        return pd.DataFrame()

    agg_dict = {}
    if 'open' in df.columns:
        agg_dict['open'] = 'first'
    if 'high' in df.columns:
        agg_dict['high'] = 'max'
    if 'low' in df.columns:
        agg_dict['low'] = 'min'
    if 'close' in df.columns:
        agg_dict['close'] = 'last'
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    # Resample OHLC
    resampled = df[available_cols].resample(timeframe).agg(agg_dict).dropna()

    return resampled


def compute_indicators_for_timeframe(
    df: pd.DataFrame,
    timeframe_suffix: str
) -> pd.DataFrame:
    """Compute key indicators for a specific timeframe.

    Focuses on low-NaN indicators to preserve training samples:
    - 5m: RSI(14), EMA(20) - RSI causes ~17% NaN
    - 15m: RSI(9), EMA(14), MACD - RSI causes ~25% NaN, MACD ~0%
    - 1h: EMA(10) only - RSI would cause 60%+ NaN

    Args:
        df: OHLC DataFrame for this timeframe
        timeframe_suffix: Suffix to add to column names ('_5m', '_15m', '_1h')

    Returns:
        DataFrame with indicator columns
    """
    result = pd.DataFrame(index=df.index)

    if 'close' not in df.columns:
        return result

    close = df['close']

    # EMA periods for each timeframe
    if timeframe_suffix == '_1h':
        ema_period = 10
    elif timeframe_suffix == '_15m':
        ema_period = 14
    else:  # 5m
        ema_period = 20

    # EMA (all timeframes - causes 0% NaN)
    result[f'ema{timeframe_suffix}'] = close.ewm(span=ema_period, adjust=False).mean()
    result[f'price_vs_ema{timeframe_suffix}'] = close / result[f'ema{timeframe_suffix}'] - 1

    # RSI only for 5m and 15m (1h RSI causes 60%+ NaN)
    if timeframe_suffix in ['_5m', '_15m']:
        rsi_period = 14 if timeframe_suffix == '_5m' else 9
        result[f'rsi{timeframe_suffix}'] = compute_rsi(close, rsi_period)

    # MACD only for 15m (causes 0% NaN, very useful indicator)
    if timeframe_suffix == '_15m':
        macd, signal, hist = compute_macd(close)
        result[f'macd{timeframe_suffix}'] = macd
        result[f'macd_signal{timeframe_suffix}'] = signal
        result[f'macd_hist{timeframe_suffix}'] = hist

    return result


def add_multi_timeframe_features(
    df: pd.DataFrame,
    ohlc_df: pd.DataFrame
) -> pd.DataFrame:
    """Add multi-timeframe indicator features.

    Resamples 1-minute OHLC data to 5m, 15m, and 1h timeframes,
    calculates indicators for each, and merges back to 1-min data.

    Args:
        df: Main DataFrame (1-min data)
        ohlc_df: Raw OHLC DataFrame (1-min)

    Returns:
        DataFrame with multi-timeframe features added
    """
    if ohlc_df is None or ohlc_df.empty:
        return df

    df = df.copy()

    # Ensure datetime for merging
    if 'datetime' not in df.columns and 'ts' in df.columns:
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')

    if 'datetime' not in df.columns:
        return df

    timeframes = [
        ('5T', '_5m'),    # 5 minutes
        ('15T', '_15m'),  # 15 minutes
        ('1H', '_1h'),    # 1 hour
    ]

    for tf_code, tf_suffix in timeframes:
        # Resample OHLC to this timeframe
        tf_ohlc = resample_ohlc(ohlc_df, tf_code)

        if len(tf_ohlc) < 20:  # Need minimum data for indicators
            continue

        # Compute indicators for this timeframe
        tf_indicators = compute_indicators_for_timeframe(tf_ohlc, tf_suffix)

        if tf_indicators.empty:
            continue

        # Reset index for merge
        tf_indicators = tf_indicators.reset_index()
        tf_indicators = tf_indicators.rename(columns={'datetime': 'tf_datetime', 'index': 'tf_datetime'})

        # Merge back to 1-min data using backward direction (no look-ahead bias)
        df = pd.merge_asof(
            df.sort_values('datetime'),
            tf_indicators.sort_values('tf_datetime'),
            left_on='datetime',
            right_on='tf_datetime',
            direction='backward'
        )

        # Drop the temporary timeframe datetime column
        df = df.drop(columns=['tf_datetime'], errors='ignore')

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


def create_features_from_ohlc(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features directly from OHLC data (for Kraken REST API data).

    This is optimized for the 1-minute candle data from Kraken REST API,
    which includes: ts, pair, open, high, low, close, volume, vwap, trades
    """
    df = ohlc_df.copy()

    # Ensure we have required columns
    required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use close as 'last' price for compatibility
    if 'last' not in df.columns:
        df['last'] = df['close']

    # Add technical indicators
    df = add_technical_features(df)

    # Add momentum features
    df = add_momentum_features(df)

    # Add volume features
    df = add_volume_features(df)

    # Add oscillator features
    df = add_oscillator_features(df)

    # Add VWAP-based features if available
    if 'vwap' in df.columns:
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        df['vwap_above'] = (df['close'] > df['vwap']).astype(int)

    # Add trade count features if available
    if 'trades' in df.columns:
        df['trades_sma_5'] = df['trades'].rolling(5).mean()
        df['trades_sma_20'] = df['trades'].rolling(20).mean()
        df['trades_ratio'] = df['trades'] / df['trades_sma_20']

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
        ohlc_cols = ['ts', 'open', 'high', 'low', 'close', 'volume', 'trades']
        ohlc_cols = [c for c in ohlc_cols if c in ohlc_df.columns]
        ohlc_subset = ohlc_df[ohlc_cols].copy()

        # Rename to avoid conflicts with ticker columns
        rename_map = {
            'high': 'high',
            'low': 'low',
            'open': 'open',
            'close': 'close',
            'volume': 'ohlc_volume',
        }
        # Only rename if column exists and would conflict
        for old, new in list(rename_map.items()):
            if old in df.columns and old in ohlc_subset.columns:
                # Drop the ticker version, we want OHLC data
                df = df.drop(columns=[old], errors='ignore')

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

    # Add oscillator features (Stochastic, MFI, ADX, OBV)
    df = add_oscillator_features(df)

    # Add multi-timeframe features (5m, 15m, 1h indicators)
    if ohlc_df is not None and not ohlc_df.empty:
        df = add_multi_timeframe_features(df, ohlc_df)

    return df


# =============================================================================
# Retracement Strategy (Drop & Recovery)
# =============================================================================

def detect_large_drops(
    df: pd.DataFrame,
    drop_threshold: float = 0.01,
    lookback: int = 1,
) -> pd.DataFrame:
    """
    Detect large price drops (red candles).

    Args:
        df: DataFrame with price data
        drop_threshold: Minimum drop to consider (0.01 = 1%)
        lookback: Number of periods to measure drop over

    Returns:
        DataFrame with drop detection columns added
    """
    df = df.copy()

    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col]

    # Calculate returns over lookback period
    df['drop_pct'] = (prices / prices.shift(lookback) - 1)

    # Detect large drops (negative returns exceeding threshold)
    df['is_large_drop'] = (df['drop_pct'] < -drop_threshold).astype(int)

    # Drop magnitude (absolute value, only for drops)
    df['drop_magnitude'] = np.where(
        df['drop_pct'] < 0,
        -df['drop_pct'],
        0
    )

    # Price at drop start and end
    df['drop_start_price'] = prices.shift(lookback)
    df['drop_end_price'] = prices

    # 50% retracement target
    df['retracement_target'] = df['drop_end_price'] + (df['drop_start_price'] - df['drop_end_price']) * 0.5

    return df


def retracement_labels(
    df: pd.DataFrame,
    drop_threshold: float = 0.01,
    max_horizon: int = 30,
    retracement_pct: float = 0.5,
    stop_loss_mult: float = 1.0,
) -> pd.Series:
    """
    Create labels for the retracement strategy.

    After a large drop, check if price retraces to target before:
    1. Hitting stop loss (drop continues by stop_loss_mult * original drop)
    2. Time expires (max_horizon periods)

    Labels:
        1 = Successful retracement (price hit 50% target)
       -1 = Failed (hit stop loss or dropped further)
        0 = Expired without hitting either / No trade signal

    Args:
        df: DataFrame with price data (must have detect_large_drops applied)
        drop_threshold: Minimum drop to trigger signal
        max_horizon: Max periods to wait for retracement
        retracement_pct: Target retracement (0.5 = 50%)
        stop_loss_mult: Stop loss as multiple of original drop

    Returns:
        Series with labels
    """
    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col].values
    n = len(prices)

    # Ensure drop detection columns exist
    if 'is_large_drop' not in df.columns:
        df = detect_large_drops(df, drop_threshold)

    is_drop = df['is_large_drop'].values
    drop_magnitude = df['drop_magnitude'].values
    drop_end_price = df['drop_end_price'].values if 'drop_end_price' in df.columns else prices
    drop_start_price = df['drop_start_price'].values if 'drop_start_price' in df.columns else prices

    labels = np.zeros(n)
    labels[:] = np.nan

    for i in range(n - max_horizon):
        # Only create label if there was a large drop
        if is_drop[i] != 1:
            labels[i] = 0  # No trade signal
            continue

        entry_price = prices[i]
        drop_size = drop_magnitude[i]

        if drop_size <= 0 or np.isnan(drop_size):
            labels[i] = 0
            continue

        # Calculate targets
        # Retracement target: recover retracement_pct of the drop
        retracement_target = entry_price * (1 + drop_size * retracement_pct)

        # Stop loss: drop continues by stop_loss_mult * original drop
        stop_loss_price = entry_price * (1 - drop_size * stop_loss_mult)

        label = 0  # Default: expired

        for j in range(1, max_horizon + 1):
            if i + j >= n:
                break

            current_price = prices[i + j]

            if np.isnan(current_price):
                continue

            # Check retracement target (win)
            if current_price >= retracement_target:
                label = 1
                break

            # Check stop loss (loss)
            if current_price <= stop_loss_price:
                label = -1
                break

        labels[i] = label

    return pd.Series(labels, index=df.index, name='retracement_label')


def analyze_retracement_stats(
    df: pd.DataFrame,
    drop_thresholds: List[float] = [0.005, 0.01, 0.015, 0.02, 0.03],
    max_horizon: int = 30,
) -> pd.DataFrame:
    """
    Analyze retracement statistics for different drop thresholds.

    Returns DataFrame with stats for each threshold:
    - Number of signals
    - Win rate (successful retracements)
    - Average time to retracement
    - Average drop magnitude
    """
    results = []

    for threshold in drop_thresholds:
        df_temp = detect_large_drops(df.copy(), drop_threshold=threshold)
        labels = retracement_labels(df_temp, drop_threshold=threshold, max_horizon=max_horizon)

        df_temp['label'] = labels

        # Only consider rows where there was a signal
        signals = df_temp[df_temp['is_large_drop'] == 1]

        if len(signals) == 0:
            continue

        n_signals = len(signals)
        n_wins = (signals['label'] == 1).sum()
        n_losses = (signals['label'] == -1).sum()
        n_expired = (signals['label'] == 0).sum()

        win_rate = n_wins / n_signals if n_signals > 0 else 0
        avg_drop = signals['drop_magnitude'].mean()

        results.append({
            'drop_threshold': f"{threshold:.1%}",
            'n_signals': n_signals,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'n_expired': n_expired,
            'win_rate': f"{win_rate:.1%}",
            'avg_drop': f"{avg_drop:.2%}",
        })

    return pd.DataFrame(results)


def add_retracement_features(df: pd.DataFrame, drop_threshold: float = 0.01) -> pd.DataFrame:
    """
    Add features related to the retracement strategy.

    Features added:
    - Recent drop patterns
    - Volatility context
    - Volume during drop
    - Distance from support levels
    """
    df = df.copy()

    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col]

    # Drop detection
    df = detect_large_drops(df, drop_threshold=drop_threshold)

    # Rolling drop statistics
    df['drops_last_10'] = df['is_large_drop'].rolling(10).sum()
    df['drops_last_30'] = df['is_large_drop'].rolling(30).sum()

    # Average drop magnitude recently
    df['avg_drop_magnitude_10'] = df['drop_magnitude'].rolling(10).mean()

    # Consecutive drops
    df['consecutive_red'] = (df['drop_pct'] < 0).astype(int)
    df['consecutive_red_count'] = df['consecutive_red'].groupby(
        (df['consecutive_red'] != df['consecutive_red'].shift()).cumsum()
    ).cumcount() + 1
    df['consecutive_red_count'] = np.where(df['consecutive_red'] == 1, df['consecutive_red_count'], 0)

    # Distance from recent high (potential support)
    df['distance_from_high_10'] = prices / prices.rolling(10).max() - 1
    df['distance_from_high_20'] = prices / prices.rolling(20).max() - 1

    # Distance from recent low
    df['distance_from_low_10'] = prices / prices.rolling(10).min() - 1
    df['distance_from_low_20'] = prices / prices.rolling(20).min() - 1

    # Volatility at drop
    df['volatility_at_drop'] = prices.pct_change().rolling(10).std()

    # Volume spike (if volume available)
    if 'volume' in df.columns:
        df['volume_vs_avg'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike_at_drop'] = df['volume_vs_avg'] * df['is_large_drop']

    # RSI at drop (oversold condition)
    if 'rsi' in df.columns:
        df['rsi_at_drop'] = df['rsi'] * df['is_large_drop']
        df['is_oversold_drop'] = ((df['rsi'] < 30) & (df['is_large_drop'] == 1)).astype(int)

    # Cleanup
    df = df.drop(columns=['consecutive_red'], errors='ignore')

    return df


# =============================================================================
# Target Creation
# =============================================================================

def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    take_profit: float = None,
    stop_loss: float = None,
    use_atr: bool = True,
    atr_tp_mult: float = 2.0,
    atr_sl_mult: float = 1.0,
    min_return: float = 0.001,
) -> pd.Series:
    """
    Create labels using the Triple Barrier Method.

    Three barriers:
    1. Upper barrier (take profit): Label = 1 if hit first
    2. Lower barrier (stop loss): Label = -1 if hit first
    3. Vertical barrier (time): Label = 0 if neither hit within horizon

    Args:
        df: DataFrame with price data
        horizon: Maximum periods to hold (vertical barrier)
        take_profit: Fixed take profit level (% as decimal, e.g., 0.02 for 2%)
        stop_loss: Fixed stop loss level (% as decimal, e.g., 0.01 for 1%)
        use_atr: If True, use ATR for dynamic barriers
        atr_tp_mult: Multiplier for ATR take profit (default 2x ATR)
        atr_sl_mult: Multiplier for ATR stop loss (default 1x ATR)
        min_return: Minimum return threshold for label=0 classification

    Returns:
        Series with labels: 1 (take profit), -1 (stop loss), 0 (expired/neutral)
    """
    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col].values
    n = len(prices)

    labels = np.zeros(n)
    labels[:] = np.nan

    # Calculate ATR if needed
    if use_atr and 'atr' in df.columns:
        atr = df['atr'].values
    elif use_atr and all(col in df.columns for col in ['high', 'low', 'close']):
        # Calculate ATR on the fly
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14).mean().values
    else:
        atr = None

    # Process each point
    for i in range(n - horizon):
        entry_price = prices[i]

        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Determine barriers
        if use_atr and atr is not None and not np.isnan(atr[i]) and atr[i] > 0:
            # Dynamic barriers based on ATR
            tp_level = entry_price * (1 + atr_tp_mult * atr[i] / entry_price)
            sl_level = entry_price * (1 - atr_sl_mult * atr[i] / entry_price)
        else:
            # Fixed barriers
            tp_pct = take_profit if take_profit is not None else 0.01  # 1% default
            sl_pct = stop_loss if stop_loss is not None else 0.005    # 0.5% default
            tp_level = entry_price * (1 + tp_pct)
            sl_level = entry_price * (1 - sl_pct)

        # Look forward through the horizon
        label = 0  # Default: expired without hitting barrier
        exit_return = 0

        for j in range(1, horizon + 1):
            if i + j >= n:
                break

            current_price = prices[i + j]

            if np.isnan(current_price):
                continue

            # Check upper barrier (take profit)
            if current_price >= tp_level:
                label = 1
                break

            # Check lower barrier (stop loss)
            if current_price <= sl_level:
                label = -1
                break

        # If no barrier hit, check final return for label
        if label == 0 and i + horizon < n:
            final_price = prices[i + horizon]
            if not np.isnan(final_price):
                exit_return = (final_price - entry_price) / entry_price
                # Use min_return threshold to classify neutral outcomes
                if exit_return > min_return:
                    label = 1
                elif exit_return < -min_return:
                    label = -1
                # else stays 0

        labels[i] = label

    return pd.Series(labels, index=df.index, name=f'tb_label_{horizon}')


def get_barrier_touches(
    df: pd.DataFrame,
    horizon: int,
    take_profit: float = None,
    stop_loss: float = None,
    use_atr: bool = True,
    atr_tp_mult: float = 2.0,
    atr_sl_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Get detailed information about barrier touches.

    Returns DataFrame with:
    - label: 1 (TP), -1 (SL), 0 (expired)
    - touch_time: periods until barrier touch
    - exit_return: actual return at exit
    - tp_level: take profit price level
    - sl_level: stop loss price level
    """
    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col].values
    n = len(prices)

    results = {
        'label': np.full(n, np.nan),
        'touch_time': np.full(n, np.nan),
        'exit_return': np.full(n, np.nan),
        'tp_level': np.full(n, np.nan),
        'sl_level': np.full(n, np.nan),
    }

    # Calculate ATR
    if use_atr and 'atr' in df.columns:
        atr = df['atr'].values
    elif use_atr and all(col in df.columns for col in ['high', 'low', 'close']):
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14).mean().values
    else:
        atr = None

    for i in range(n - horizon):
        entry_price = prices[i]

        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Determine barriers
        if use_atr and atr is not None and not np.isnan(atr[i]) and atr[i] > 0:
            tp_level = entry_price * (1 + atr_tp_mult * atr[i] / entry_price)
            sl_level = entry_price * (1 - atr_sl_mult * atr[i] / entry_price)
        else:
            tp_pct = take_profit if take_profit is not None else 0.01
            sl_pct = stop_loss if stop_loss is not None else 0.005
            tp_level = entry_price * (1 + tp_pct)
            sl_level = entry_price * (1 - sl_pct)

        results['tp_level'][i] = tp_level
        results['sl_level'][i] = sl_level

        label = 0
        touch_time = horizon
        exit_price = prices[min(i + horizon, n - 1)]

        for j in range(1, horizon + 1):
            if i + j >= n:
                break

            current_price = prices[i + j]

            if np.isnan(current_price):
                continue

            if current_price >= tp_level:
                label = 1
                touch_time = j
                exit_price = current_price
                break

            if current_price <= sl_level:
                label = -1
                touch_time = j
                exit_price = current_price
                break

        results['label'][i] = label
        results['touch_time'][i] = touch_time
        results['exit_return'][i] = (exit_price - entry_price) / entry_price if not np.isnan(exit_price) else np.nan

    return pd.DataFrame(results, index=df.index)


def create_targets(
    df: pd.DataFrame,
    horizons: List[int] = [5, 15, 30, 60],
    use_triple_barrier: bool = False,
    tb_take_profit: float = None,
    tb_stop_loss: float = None,
    tb_use_atr: bool = True,
) -> pd.DataFrame:
    """
    Create forward return targets for different horizons.

    Args:
        df: DataFrame with price data
        horizons: List of forward-looking periods
        use_triple_barrier: If True, also create triple barrier labels
        tb_take_profit: Fixed take profit for triple barrier (% as decimal)
        tb_stop_loss: Fixed stop loss for triple barrier (% as decimal)
        tb_use_atr: Use ATR for dynamic barriers

    Returns:
        DataFrame with target columns added
    """
    df = df.copy()

    price_col = 'close' if 'close' in df.columns else 'last'
    prices = df[price_col]

    for h in horizons:
        # Forward return (simple)
        df[f'target_{h}m'] = prices.shift(-h) / prices - 1

        # Direction (1 = up, 0 = down)
        df[f'direction_{h}m'] = (df[f'target_{h}m'] > 0).astype(int)

        # Triple barrier labels
        if use_triple_barrier:
            df[f'tb_label_{h}m'] = triple_barrier_labels(
                df,
                horizon=h,
                take_profit=tb_take_profit,
                stop_loss=tb_stop_loss,
                use_atr=tb_use_atr,
            )

            # Also create binary classification target (1=profit, 0=loss/neutral)
            df[f'tb_binary_{h}m'] = (df[f'tb_label_{h}m'] == 1).astype(int)

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


@dataclass
class ClassificationResult:
    """Results from classification model training."""
    name: str
    model: object
    accuracy_scores: List[float]
    precision_scores: List[float]
    recall_scores: List[float]
    f1_scores: List[float]
    feature_importance: Dict[str, float]
    class_distribution: Dict[int, int]


def train_classification_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_splits: int = 5,
) -> List[ClassificationResult]:
    """
    Train classification models for triple barrier labels.

    Args:
        df: DataFrame with features and target
        target_col: Column name for triple barrier labels (-1, 0, 1)
        feature_cols: List of feature column names
        n_splits: Number of cross-validation splits

    Returns:
        List of ClassificationResult objects
    """
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values

    # Remove rows with NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y_original = y[mask].astype(int)

    # Remap labels to sequential integers starting from 0 for XGBoost compatibility
    # This handles cases where not all classes are present after NaN filtering
    unique_labels = sorted(np.unique(y_original))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
    reverse_label_map = {idx: orig for orig, idx in label_map.items()}
    y = np.array([label_map[label] for label in y_original])

    if len(X) < 100:
        print(f"Warning: Only {len(X)} samples available for training")
        return []

    # Class distribution (show original labels)
    unique, counts = np.unique(y_original, return_counts=True)
    class_dist = dict(zip(unique.astype(int), counts.astype(int)))

    print(f"\nTraining CLASSIFICATION with {len(X)} samples, {len(feature_cols)} features")
    print(f"Class distribution: {class_dist}")
    for label, count in class_dist.items():
        label_name = {-1: 'Stop Loss', 0: 'Expired', 1: 'Take Profit'}.get(label, str(label))
        print(f"  {label_name}: {count} ({count/len(y)*100:.1f}%)")

    # Models to train
    models = {
        'RF_Classifier': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        ),
    }

    if HAS_XGB:
        models['XGB_Classifier'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=10,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

    if HAS_LGB:
        models['LGB_Classifier'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores_list = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            # Use weighted average for multiclass
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            acc_scores.append(acc)
            prec_scores.append(prec)
            rec_scores.append(rec)
            f1_scores_list.append(f1)

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_cols, model.feature_importances_))
        else:
            importance = {}

        result = ClassificationResult(
            name=name,
            model=model,
            accuracy_scores=acc_scores,
            precision_scores=prec_scores,
            recall_scores=rec_scores,
            f1_scores=f1_scores_list,
            feature_importance=importance,
            class_distribution=class_dist
        )
        results.append(result)

        print(f"  Accuracy = {np.mean(acc_scores):.2%} (+/- {np.std(acc_scores):.2%})")
        print(f"  Precision = {np.mean(prec_scores):.2%} (+/- {np.std(prec_scores):.2%})")
        print(f"  Recall = {np.mean(rec_scores):.2%} (+/- {np.std(rec_scores):.2%})")
        print(f"  F1 Score = {np.mean(f1_scores_list):.2%} (+/- {np.std(f1_scores_list):.2%})")

    # Print detailed classification report for best model
    if results:
        best = max(results, key=lambda r: np.mean(r.f1_scores))
        print(f"\nBest Classifier: {best.name}")

        # Final predictions on last fold (remap back to original labels)
        y_pred_remapped = best.model.predict(X)
        y_pred_final = np.array([reverse_label_map.get(p, p) for p in y_pred_remapped])
        print("\nClassification Report (Full Dataset):")
        print(classification_report(
            y_original, y_pred_final,
            labels=[-1, 0, 1],
            target_names=['Stop Loss (-1)', 'Expired (0)', 'Take Profit (1)'],
            zero_division=0
        ))

    return results


@dataclass
class BacktestResult:
    """Results from backtesting a strategy."""
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def backtest_triple_barrier(
    df: pd.DataFrame,
    predictions: np.ndarray,
    price_col: str = 'last',
    hold_periods: int = 30,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """
    Backtest strategy using triple barrier predictions.

    Args:
        df: DataFrame with price data
        predictions: Array of predicted labels (1=long, -1=avoid, 0=neutral)
        price_col: Price column name
        hold_periods: How long to hold position after entry
        transaction_cost: Cost per trade (as decimal)

    Returns:
        BacktestResult with equity curve and metrics
    """
    prices = df[price_col].values
    n = len(prices)

    # Align predictions
    predictions = predictions[:n]

    equity = [1.0]
    returns_list = []
    n_trades = 0
    wins = 0

    i = 0
    while i < n - hold_periods:
        if predictions[i] == 1:  # Only trade when model predicts take profit
            entry_price = prices[i]
            exit_price = prices[min(i + hold_periods, n - 1)]

            trade_return = (exit_price - entry_price) / entry_price
            trade_return -= transaction_cost * 2  # Entry + exit cost

            equity.append(equity[-1] * (1 + trade_return))
            returns_list.append(trade_return)

            n_trades += 1
            if trade_return > 0:
                wins += 1

            i += hold_periods  # Skip ahead after trade
        else:
            equity.append(equity[-1])
            returns_list.append(0)
            i += 1

    # Pad equity to match dataframe
    while len(equity) < n:
        equity.append(equity[-1])
    while len(returns_list) < n - 1:
        returns_list.append(0)

    equity_series = pd.Series(equity[:n], index=df.index[:n])
    returns_series = pd.Series(returns_list)

    # Metrics
    total_return = equity[-1] - 1

    non_zero_returns = [r for r in returns_list if r != 0]
    if len(non_zero_returns) > 0 and np.std(non_zero_returns) > 0:
        sharpe = np.mean(non_zero_returns) / np.std(non_zero_returns) * np.sqrt(252 * 24)  # Annualized
    else:
        sharpe = 0

    # Max drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    win_rate = wins / n_trades if n_trades > 0 else 0

    return BacktestResult(
        equity_curve=equity_series,
        returns=returns_series,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=n_trades
    )


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


def backtest_retracement(
    df: pd.DataFrame,
    predictions: np.ndarray,
    drop_threshold: float = 0.005,
    retracement_pct: float = 0.5,
    stop_loss_mult: float = 1.0,
    max_hold: int = 30,
    transaction_cost: float = 0.0026,  # Kraken taker fee
    price_col: str = 'last',
) -> BacktestResult:
    """
    Backtest the retracement strategy.

    Only enters when:
    1. There was a large drop (is_large_drop == 1)
    2. Model predicts successful retracement (prediction == 1)

    Exit when:
    - Price hits retracement target (take profit)
    - Price hits stop loss
    - Max hold periods reached

    Args:
        df: DataFrame with price data and drop detection columns
        predictions: Model predictions (1=enter, 0/-1=skip)
        drop_threshold: Minimum drop to trigger signal
        retracement_pct: Target retracement (0.5 = 50%)
        stop_loss_mult: Stop loss as multiple of original drop
        max_hold: Maximum periods to hold position
        transaction_cost: Cost per trade (entry + exit)
        price_col: Price column name

    Returns:
        BacktestResult with equity curve and metrics
    """
    # Ensure drop detection columns exist
    if 'is_large_drop' not in df.columns:
        df = detect_large_drops(df, drop_threshold=drop_threshold)

    prices = df[price_col].values
    is_drop = df['is_large_drop'].values
    drop_magnitude = df['drop_magnitude'].values

    n = len(prices)
    predictions = predictions[:n]

    equity = [1.0]
    returns_list = []
    trades = []
    n_trades = 0
    wins = 0

    i = 0
    while i < n - max_hold:
        # Check if we should enter: large drop AND model predicts success
        if is_drop[i] == 1 and predictions[i] == 1:
            entry_price = prices[i]
            drop_size = drop_magnitude[i]

            if drop_size <= 0 or np.isnan(drop_size):
                i += 1
                continue

            # Calculate targets
            take_profit = entry_price * (1 + drop_size * retracement_pct)
            stop_loss = entry_price * (1 - drop_size * stop_loss_mult)

            # Simulate trade
            exit_price = None
            exit_reason = 'expired'

            for j in range(1, max_hold + 1):
                if i + j >= n:
                    break

                current_price = prices[i + j]

                if current_price >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
                elif current_price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break

            # If no exit, use last price
            if exit_price is None:
                exit_price = prices[min(i + max_hold, n - 1)]

            # Calculate return
            trade_return = (exit_price - entry_price) / entry_price
            trade_return -= transaction_cost * 2  # Entry + exit

            equity.append(equity[-1] * (1 + trade_return))
            returns_list.append(trade_return)

            n_trades += 1
            if trade_return > 0:
                wins += 1

            trades.append({
                'entry_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': trade_return,
                'reason': exit_reason,
                'drop_size': drop_size,
            })

            # Skip ahead after trade
            i += max_hold
        else:
            i += 1

    # Pad equity curve
    while len(equity) < n:
        equity.append(equity[-1])

    equity_series = pd.Series(equity[:n], index=df.index[:n])
    returns_series = pd.Series(returns_list) if returns_list else pd.Series([0])

    # Metrics
    total_return = equity[-1] - 1

    if len(returns_list) > 0 and np.std(returns_list) > 0:
        sharpe = np.mean(returns_list) / np.std(returns_list) * np.sqrt(252 * 24)
    else:
        sharpe = 0

    # Max drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    win_rate = wins / n_trades if n_trades > 0 else 0

    return BacktestResult(
        equity_curve=equity_series,
        returns=returns_series,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=n_trades
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
        'target_', 'direction_', 'tb_label_', 'tb_binary_',
        'retracement_label',  # Exclude to avoid data leakage
        'datetime', 'ts', 'pair', 'bids', 'asks', '_json'
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
    parser.add_argument('--source', type=str, default='websocket', choices=['websocket', 'kraken'],
                        help='Data source: websocket (captured) or kraken (REST API candles)')
    parser.add_argument('--ohlc-interval', type=int, default=1,
                        help='OHLC candle interval in minutes for Kraken source (1, 5, 15, 60, 1440)')
    parser.add_argument('--target', type=str, default='5m', help='Target horizon (5m, 15m, 30m, 1h, 4h, 1d)')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save-model', type=str, help='Save best model to path')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory for plots')

    # Triple Barrier arguments
    parser.add_argument('--triple-barrier', action='store_true',
                        help='Use Triple Barrier labeling method for classification')
    parser.add_argument('--tb-take-profit', type=float, default=None,
                        help='Fixed take profit %% (e.g., 0.01 for 1%%). Default: use ATR')
    parser.add_argument('--tb-stop-loss', type=float, default=None,
                        help='Fixed stop loss %% (e.g., 0.005 for 0.5%%). Default: use ATR')
    parser.add_argument('--tb-atr-tp', type=float, default=2.0,
                        help='ATR multiplier for take profit (default: 2.0)')
    parser.add_argument('--tb-atr-sl', type=float, default=1.0,
                        help='ATR multiplier for stop loss (default: 1.0)')

    # Retracement Strategy arguments
    parser.add_argument('--retracement', action='store_true',
                        help='Use Retracement strategy (buy after large drops)')
    parser.add_argument('--drop-threshold', type=float, default=0.005,
                        help='Minimum drop to trigger signal (0.005 = 0.5%%)')
    parser.add_argument('--retracement-pct', type=float, default=0.5,
                        help='Target retracement percentage (0.5 = 50%%)')
    parser.add_argument('--max-hold', type=int, default=30,
                        help='Maximum periods to hold position (default: 30)')
    parser.add_argument('--analyze-drops', action='store_true',
                        help='Show drop analysis statistics before training')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Parse target horizon
    target_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    if args.target not in target_map:
        print(f"Invalid target: {args.target}. Use one of: {list(target_map.keys())}")
        return
    target_horizon = target_map[args.target]

    # For OHLC data, target horizon is in candles, not minutes
    # Adjust based on ohlc-interval
    candles_per_target = target_horizon  # default for 1-minute data
    if args.source == 'kraken' and args.ohlc_interval > 1:
        # Convert target from minutes to number of candles
        candles_per_target = target_horizon // args.ohlc_interval
        if candles_per_target < 1:
            candles_per_target = 1
        print(f"Note: With {args.ohlc_interval}m candles, {args.target} = {candles_per_target} candles forward")

    # Determine target column and mode
    # For non-1-minute data, use candle-based horizon
    horizon_suffix = candles_per_target if (args.source == 'kraken' and args.ohlc_interval > 1) else target_horizon
    if args.retracement:
        target_col = 'retracement_label'
        strategy_mode = 'retracement'
    elif args.triple_barrier:
        target_col = f'tb_label_{horizon_suffix}m'
        strategy_mode = 'triple_barrier'
    else:
        target_col = f'target_{horizon_suffix}m'
        strategy_mode = 'regression'

    print(f"\n{'='*60}")
    print(f"ML Analysis for {args.pair}")

    if args.retracement:
        print(f"Strategy: RETRACEMENT (Drop & Recovery)")
        print(f"Drop threshold: {args.drop_threshold:.2%}")
        print(f"Retracement target: {args.retracement_pct:.0%}")
        print(f"Max hold: {args.max_hold} periods")
    elif args.triple_barrier:
        print(f"Target: {args.target} forward (Triple Barrier)")
        if args.tb_take_profit and args.tb_stop_loss:
            print(f"Barriers: TP={args.tb_take_profit:.2%}, SL={args.tb_stop_loss:.2%}")
        else:
            print(f"Barriers: ATR-based (TP={args.tb_atr_tp}x, SL={args.tb_atr_sl}x)")
    else:
        print(f"Target: {args.target} forward return")

    print(f"{'='*60}\n")

    # Load data based on source
    print("Loading data...")

    if args.source == 'kraken':
        # Load from Kraken REST API parquet files
        kraken_df = load_kraken_ohlc_data(data_dir, args.pair, interval=args.ohlc_interval)

        if kraken_df.empty:
            print(f"No Kraken REST data found for {args.pair}")
            available = list_kraken_pairs(data_dir)
            if available:
                print(f"Available pairs: {available}")
            return

        # Create features directly from OHLC
        print("\nCreating features from Kraken OHLC data...")
        df = create_features_from_ohlc(kraken_df)

    else:
        # Load from WebSocket captured data (original method)
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

    # Create targets based on strategy mode
    print("Creating targets...")

    if args.retracement:
        # Add retracement-specific features
        df = add_retracement_features(df, drop_threshold=args.drop_threshold)

        # Show drop analysis if requested
        if args.analyze_drops:
            print("\n" + "="*60)
            print("DROP ANALYSIS")
            print("="*60)
            stats = analyze_retracement_stats(
                df,
                drop_thresholds=[0.001, 0.002, 0.003, 0.005, 0.01, 0.02],
                max_horizon=args.max_hold
            )
            print(stats.to_string(index=False))
            print()

        # Create retracement labels
        df['retracement_label'] = retracement_labels(
            df,
            drop_threshold=args.drop_threshold,
            max_horizon=args.max_hold,
            retracement_pct=args.retracement_pct,
        )

        # Show label distribution
        signals = df[df['is_large_drop'] == 1]
        if len(signals) > 0:
            print(f"\nRetracement signals found: {len(signals)}")
            wins = (signals['retracement_label'] == 1).sum()
            losses = (signals['retracement_label'] == -1).sum()
            expired = (signals['retracement_label'] == 0).sum()
            print(f"  ✅ Successful retracements: {wins} ({wins/len(signals)*100:.1f}%)")
            print(f"  ❌ Stop loss hit: {losses} ({losses/len(signals)*100:.1f}%)")
            print(f"  ⏱️  Expired: {expired} ({expired/len(signals)*100:.1f}%)")
    else:
        # Standard targets (regression or triple barrier)
        use_atr = args.tb_take_profit is None or args.tb_stop_loss is None

        # Adjust horizons based on OHLC interval
        # For 1-minute candles: [5, 15, 30, 60] = 5m, 15m, 30m, 1h
        # For 60-minute candles: [1, 2, 4, 6] = 1h, 2h, 4h, 6h
        # For 1440-minute (daily) candles: [1, 2, 5, 7] = 1d, 2d, 5d, 1w
        if args.source == 'kraken' and args.ohlc_interval > 1:
            if args.ohlc_interval == 60:  # 1h candles
                horizons = [1, 2, 4, 6]  # 1h, 2h, 4h, 6h
                horizon_labels = ['1h', '2h', '4h', '6h']
            elif args.ohlc_interval == 1440:  # daily candles
                horizons = [1, 2, 5, 7]  # 1d, 2d, 5d, 1w
                horizon_labels = ['1d', '2d', '5d', '1w']
            else:
                # Convert minutes to candles
                horizons = [max(1, h // args.ohlc_interval) for h in [5, 15, 30, 60]]
                horizon_labels = [f'{h}' for h in horizons]
            print(f"Using horizons: {horizons} ({horizon_labels})")
        else:
            horizons = [5, 15, 30, 60]

        df = create_targets(
            df,
            horizons=horizons,
            use_triple_barrier=args.triple_barrier,
            tb_take_profit=args.tb_take_profit,
            tb_stop_loss=args.tb_stop_loss,
            tb_use_atr=use_atr,
        )

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
    if args.retracement:
        print("TRAINING CLASSIFICATION MODELS (Retracement Strategy)")
    elif args.triple_barrier:
        print("TRAINING CLASSIFICATION MODELS (Triple Barrier)")
    else:
        print("TRAINING REGRESSION MODELS")
    print("="*60)

    if args.retracement or args.triple_barrier:
        # Train classification models
        clf_results = train_classification_models(df, target_col, feature_cols, n_splits=5)

        if not clf_results:
            print("No classification models trained successfully")
            return

        # Find best classifier
        best_clf = max(clf_results, key=lambda r: np.mean(r.f1_scores))
        print(f"\nBest classifier: {best_clf.name}")

        # Print top features
        if best_clf.feature_importance:
            print(f"\nTop 10 Features ({best_clf.name}):")
            sorted_features = sorted(
                best_clf.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for i, (feat, imp) in enumerate(sorted_features, 1):
                print(f"  {i:2}. {feat:30} {imp:.4f}")

        # Store for later use
        best_result = best_clf
        results = clf_results
        is_classification = True
    else:
        # Train regression models
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

        is_classification = False

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

    if is_classification:
        if args.retracement:
            # Use retracement backtest
            backtest = backtest_retracement(
                df_clean,
                predictions,
                drop_threshold=args.drop_threshold,
                retracement_pct=args.retracement_pct,
                stop_loss_mult=1.0,
                max_hold=args.max_hold,
                transaction_cost=0.0026,  # Kraken taker fee
                price_col='last',
            )
        else:
            # Use triple barrier backtest
            backtest = backtest_triple_barrier(
                df_clean,
                predictions,
                price_col='last',
                hold_periods=target_horizon,
                transaction_cost=0.001
            )
    else:
        # Use standard regression backtest
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
        model_data = {
            'model': best_result.model,
            'feature_cols': feature_cols,
            'pair': args.pair,
            'target': args.target,
            'is_classification': is_classification,
            'triple_barrier': args.triple_barrier,
        }
        if args.triple_barrier:
            model_data['tb_config'] = {
                'take_profit': args.tb_take_profit,
                'stop_loss': args.tb_stop_loss,
                'atr_tp_mult': args.tb_atr_tp,
                'atr_sl_mult': args.tb_atr_sl,
            }
        if args.retracement:
            model_data['retracement'] = True
            model_data['retracement_config'] = {
                'drop_threshold': args.drop_threshold,
                'retracement_pct': args.retracement_pct,
                'max_hold': args.max_hold,
            }
        joblib.dump(model_data, model_path)
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

        # Predictions vs actual (only for regression)
        if not is_classification:
            y_actual = df_clean[target_col].values
            plot_predictions_vs_actual(
                y_actual,
                predictions,
                best_result.name,
                save_path=output_dir / f"{args.pair.replace('/', '_')}_predictions.png"
            )
        else:
            # For classification, plot confusion matrix
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt

            y_actual = df_clean[target_col].values
            mask = ~np.isnan(y_actual)
            y_actual_clean = y_actual[mask].astype(int)
            predictions_clean = predictions[mask]

            cm = confusion_matrix(y_actual_clean, predictions_clean, labels=[-1, 0, 1])
            fig, ax = plt.subplots(figsize=(8, 6))

            # Different labels for retracement vs triple barrier
            if args.retracement:
                labels = ['Stop Loss', 'Expired', 'Retracement']
                title_suffix = 'Retracement Strategy'
            else:
                labels = ['Stop Loss', 'Expired', 'Take Profit']
                title_suffix = 'Triple Barrier'

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{best_result.name} - Confusion Matrix ({title_suffix})')
            plt.tight_layout()
            save_path = output_dir / f"{args.pair.replace('/', '_')}_confusion_matrix.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
            plt.show()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
