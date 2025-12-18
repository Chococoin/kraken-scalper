#!/usr/bin/env python3
"""AI Advisor Service for Paper Trading

Loads trained XGBoost model, generates predictions from real-time market data,
and stores signals/trades in MongoDB.

Usage:
    python3 scripts/ai_advisor.py --model models/btc_retracement.joblib --pair BTC/USD
"""

import argparse
import hashlib
import hmac
import base64
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from pymongo import MongoClient
from pymongo.collection import Collection


# Kraken API endpoints
KRAKEN_API_URL = "https://api.kraken.com"


def get_kraken_signature(urlpath: str, data: dict, secret: str) -> str:
    """Generate Kraken API signature."""
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def kraken_request(endpoint: str, data: dict = None, api_key: str = None, api_secret: str = None) -> dict:
    """Make Kraken API request."""
    url = f"{KRAKEN_API_URL}{endpoint}"

    if data is None:
        data = {}

    if api_key and api_secret:
        data['nonce'] = str(int(time.time() * 1000))
        headers = {
            'API-Key': api_key,
            'API-Sign': get_kraken_signature(endpoint, data, api_secret)
        }
        response = requests.post(url, headers=headers, data=data)
    else:
        response = requests.get(url, params=data)

    result = response.json()
    if result.get('error'):
        raise Exception(f"Kraken API error: {result['error']}")
    return result.get('result', {})


def get_ticker(pair: str) -> dict:
    """Get current ticker data for a pair."""
    # Convert pair format (BTC/USD -> XBTUSD)
    kraken_pair = pair.replace('/', '').replace('BTC', 'XBT')
    result = kraken_request('/0/public/Ticker', {'pair': kraken_pair})

    if not result:
        return {}

    # Get first result key
    key = list(result.keys())[0]
    data = result[key]

    return {
        'pair': pair,
        'bid': float(data['b'][0]),
        'ask': float(data['a'][0]),
        'last': float(data['c'][0]),
        'volume': float(data['v'][1]),  # 24h volume
        'vwap': float(data['p'][1]),    # 24h vwap
        'high': float(data['h'][1]),    # 24h high
        'low': float(data['l'][1]),     # 24h low
        'open': float(data['o']),
    }


def get_ohlc(pair: str, interval: int = 1) -> pd.DataFrame:
    """Get OHLC data for a pair."""
    kraken_pair = pair.replace('/', '').replace('BTC', 'XBT')
    result = kraken_request('/0/public/OHLC', {'pair': kraken_pair, 'interval': interval})

    if not result:
        return pd.DataFrame()

    key = list(result.keys())[0]
    data = result[key]

    df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    df['ts'] = pd.to_numeric(df['ts']) * 1000  # Convert to ms
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
        df[col] = pd.to_numeric(df[col])

    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
    """Compute Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def create_features(ticker: dict, ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Create feature DataFrame from ticker and OHLC data."""
    if ohlc_df.empty:
        return pd.DataFrame()

    # Start with ticker data
    features = {
        'bid': ticker['bid'],
        'ask': ticker['ask'],
        'last': ticker['last'],
        'vwap': ticker['vwap'],
        'volume': ticker['volume'],
        'high': ticker['high'],
        'low': ticker['low'],
        'open': ticker['open'],
    }

    # Add OHLC-derived features
    close = ohlc_df['close']
    high = ohlc_df['high']
    low = ohlc_df['low']

    # Use latest values for calculations
    if len(close) >= 14:
        features['rsi'] = compute_rsi(close, 14).iloc[-1]

    if len(close) >= 26:
        macd, macd_signal, macd_hist = compute_macd(close)
        features['macd'] = macd.iloc[-1]
        features['macd_signal'] = macd_signal.iloc[-1]
        features['macd_hist'] = macd_hist.iloc[-1]

    if len(close) >= 20:
        bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
        features['bb_upper'] = bb_upper.iloc[-1]
        features['bb_lower'] = bb_lower.iloc[-1]
        features['bb_mid'] = bb_mid.iloc[-1]
        features['bb_pct'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # EMAs
    if len(close) >= 20:
        features['ema_20'] = close.ewm(span=20).mean().iloc[-1]
    if len(close) >= 50:
        features['ema_50'] = close.ewm(span=50).mean().iloc[-1]

    # Price changes
    features['change_pct'] = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

    # Spread
    features['spread'] = ticker['ask'] - ticker['bid']
    features['spread_pct'] = features['spread'] / ticker['last'] * 100

    return pd.DataFrame([features])


class MongoDBStore:
    """MongoDB storage for AI signals and paper trades."""

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "scalper_ai"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.signals: Collection = self.db['signals']
        self.paper_trades: Collection = self.db['paper_trades']
        print(f"Connected to MongoDB: {db_name}")

    def store_signal(self, signal: dict) -> str:
        """Store an AI signal and return its ID."""
        result = self.signals.insert_one(signal)
        return str(result.inserted_id)

    def store_paper_trade(self, trade: dict) -> str:
        """Store a paper trade and return its ID."""
        result = self.paper_trades.insert_one(trade)
        return str(result.inserted_id)

    def get_open_trades(self, pair: str = None) -> list:
        """Get open paper trades."""
        query = {'status': 'open'}
        if pair:
            query['pair'] = pair
        return list(self.paper_trades.find(query))

    def close_trade(self, trade_id, exit_price: float, status: str = 'closed'):
        """Close a paper trade with exit price and PnL."""
        from bson import ObjectId

        trade = self.paper_trades.find_one({'_id': ObjectId(trade_id)})
        if not trade:
            return

        # Calculate PnL
        if trade['side'] == 'buy':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']

        pnl_pct = pnl / trade['cost'] * 100

        self.paper_trades.update_one(
            {'_id': ObjectId(trade_id)},
            {'$set': {
                'status': status,
                'exit_price': exit_price,
                'exit_time': datetime.now(timezone.utc),
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }}
        )

        print(f"  Closed trade: PnL ${pnl:.2f} ({pnl_pct:.2f}%)")

    def get_stats(self) -> dict:
        """Get trading statistics."""
        closed = list(self.paper_trades.find({'status': {'$ne': 'open'}}))

        if not closed:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }

        total = len(closed)
        winners = sum(1 for t in closed if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in closed)

        return {
            'total_trades': total,
            'winning_trades': winners,
            'losing_trades': total - winners,
            'win_rate': winners / total * 100 if total > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total if total > 0 else 0
        }


class AIAdvisor:
    """AI Advisor that generates trading signals using XGBoost model."""

    def __init__(self, model_path: str, mongo_store: MongoDBStore):
        self.mongo = mongo_store

        # Load model
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.pair = model_data.get('pair', 'BTC/USD')
        self.is_classification = model_data.get('is_classification', True)
        self.model_version = Path(model_path).stem

        print(f"  Model: {type(self.model).__name__}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Pair: {self.pair}")

        # Trading parameters
        self.min_confidence = 0.6  # Minimum confidence to act
        self.position_size = 100.0  # USD per trade
        self.stop_loss_pct = 0.02   # 2% stop loss
        self.take_profit_pct = 0.03  # 3% take profit

    def generate_signal(self, features_df: pd.DataFrame) -> dict:
        """Generate trading signal from features."""
        # Ensure we have all required features
        missing = set(self.feature_cols) - set(features_df.columns)
        for col in missing:
            features_df[col] = 0  # Fill missing with 0

        # Align columns
        X = features_df[self.feature_cols].values

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Predict
        if self.is_classification:
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Map prediction to signal
            if prediction == 1:  # Take profit / Buy signal
                signal_type = 'buy'
                confidence = probabilities[list(self.model.classes_).index(1)]
            elif prediction == -1:  # Stop loss / Sell signal
                signal_type = 'sell'
                confidence = probabilities[list(self.model.classes_).index(-1)]
            else:  # Hold
                signal_type = 'hold'
                confidence = probabilities[list(self.model.classes_).index(0)]
        else:
            # Regression - interpret as direction
            prediction = self.model.predict(X)[0]
            if prediction > 0.005:
                signal_type = 'buy'
                confidence = min(abs(prediction) * 100, 1.0)
            elif prediction < -0.005:
                signal_type = 'sell'
                confidence = min(abs(prediction) * 100, 1.0)
            else:
                signal_type = 'hold'
                confidence = 0.5

        return {
            'signal': signal_type,
            'confidence': float(confidence),
            'prediction': int(prediction) if self.is_classification else float(prediction)
        }

    def get_indicators(self, ticker: dict, features_df: pd.DataFrame) -> dict:
        """Extract indicator values for storage."""
        return {
            'price': ticker['last'],
            'rsi': float(features_df['rsi'].iloc[0]) if 'rsi' in features_df.columns else None,
            'macd': float(features_df['macd'].iloc[0]) if 'macd' in features_df.columns else None,
            'macd_signal': float(features_df['macd_signal'].iloc[0]) if 'macd_signal' in features_df.columns else None,
            'macd_hist': float(features_df['macd_hist'].iloc[0]) if 'macd_hist' in features_df.columns else None,
            'ema_20': float(features_df['ema_20'].iloc[0]) if 'ema_20' in features_df.columns else None,
            'ema_50': float(features_df['ema_50'].iloc[0]) if 'ema_50' in features_df.columns else None,
            'bb_upper': float(features_df['bb_upper'].iloc[0]) if 'bb_upper' in features_df.columns else None,
            'bb_lower': float(features_df['bb_lower'].iloc[0]) if 'bb_lower' in features_df.columns else None,
            'volume': ticker['volume'],
            'bid_ask_spread': ticker['ask'] - ticker['bid'],
        }

    def check_stop_loss_take_profit(self, ticker: dict):
        """Check open trades for stop loss / take profit."""
        current_price = ticker['last']
        open_trades = self.mongo.get_open_trades(self.pair)

        for trade in open_trades:
            entry = trade['entry_price']

            if trade['side'] == 'buy':
                pnl_pct = (current_price - entry) / entry
                if pnl_pct <= -self.stop_loss_pct:
                    print(f"  STOP LOSS triggered at ${current_price:.2f}")
                    self.mongo.close_trade(trade['_id'], current_price, 'stop_loss')
                elif pnl_pct >= self.take_profit_pct:
                    print(f"  TAKE PROFIT triggered at ${current_price:.2f}")
                    self.mongo.close_trade(trade['_id'], current_price, 'take_profit')
            else:  # sell
                pnl_pct = (entry - current_price) / entry
                if pnl_pct <= -self.stop_loss_pct:
                    print(f"  STOP LOSS triggered at ${current_price:.2f}")
                    self.mongo.close_trade(trade['_id'], current_price, 'stop_loss')
                elif pnl_pct >= self.take_profit_pct:
                    print(f"  TAKE PROFIT triggered at ${current_price:.2f}")
                    self.mongo.close_trade(trade['_id'], current_price, 'take_profit')

    def run_once(self) -> dict:
        """Run one prediction cycle."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking {self.pair}...")

        # Get market data
        ticker = get_ticker(self.pair)
        if not ticker:
            print("  Failed to get ticker data")
            return {}

        ohlc = get_ohlc(self.pair, interval=1)
        if ohlc.empty:
            print("  Failed to get OHLC data")
            return {}

        print(f"  Price: ${ticker['last']:.2f} | Bid: ${ticker['bid']:.2f} | Ask: ${ticker['ask']:.2f}")

        # Check stop loss / take profit
        self.check_stop_loss_take_profit(ticker)

        # Create features
        features_df = create_features(ticker, ohlc)
        if features_df.empty:
            print("  Failed to create features")
            return {}

        # Generate signal
        signal_data = self.generate_signal(features_df)
        indicators = self.get_indicators(ticker, features_df)

        print(f"  Signal: {signal_data['signal'].upper()} (confidence: {signal_data['confidence']*100:.1f}%)")

        # Store signal
        signal_doc = {
            'timestamp': datetime.now(timezone.utc),
            'pair': self.pair,
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'model_version': self.model_version,
            'indicators': indicators,
            'executed': False,
            'execution': None
        }
        signal_id = self.mongo.store_signal(signal_doc)

        # Execute if confidence is high enough
        if signal_data['confidence'] >= self.min_confidence and signal_data['signal'] != 'hold':
            open_trades = self.mongo.get_open_trades(self.pair)

            # Don't open if we already have a position
            if len(open_trades) == 0:
                quantity = self.position_size / ticker['last']

                trade_doc = {
                    'signal_id': signal_id,
                    'timestamp': datetime.now(timezone.utc),
                    'pair': self.pair,
                    'side': signal_data['signal'],
                    'entry_price': ticker['last'],
                    'quantity': quantity,
                    'cost': self.position_size,
                    'status': 'open',
                    'entry_confidence': signal_data['confidence'],
                    'entry_indicators': indicators
                }
                trade_id = self.mongo.store_paper_trade(trade_doc)

                print(f"  EXECUTED: {signal_data['signal'].upper()} {quantity:.6f} {self.pair} @ ${ticker['last']:.2f}")

                # Update signal as executed
                self.mongo.signals.update_one(
                    {'_id': signal_id},
                    {'$set': {
                        'executed': True,
                        'execution': {
                            'executed_at': datetime.now(timezone.utc),
                            'side': signal_data['signal'],
                            'price': ticker['last'],
                            'quantity': quantity,
                            'cost': self.position_size,
                            'paper_trade': True
                        }
                    }}
                )
            else:
                print(f"  Skipped: Already have {len(open_trades)} open position(s)")

        return signal_data


def main():
    parser = argparse.ArgumentParser(description='AI Advisor for Paper Trading')
    parser.add_argument('--model', type=str, default='models/btc_retracement.joblib',
                       help='Path to trained model')
    parser.add_argument('--pair', type=str, default='BTC/USD',
                       help='Trading pair')
    parser.add_argument('--interval', type=int, default=60,
                       help='Prediction interval in seconds')
    parser.add_argument('--mongo-uri', type=str, default='mongodb://localhost:27017',
                       help='MongoDB connection URI')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    args = parser.parse_args()

    # Connect to MongoDB
    mongo = MongoDBStore(args.mongo_uri)

    # Create advisor
    advisor = AIAdvisor(args.model, mongo)
    advisor.pair = args.pair

    print(f"\nAI Advisor started")
    print(f"  Pair: {args.pair}")
    print(f"  Interval: {args.interval}s")
    print(f"  Min confidence: {advisor.min_confidence*100:.0f}%")
    print(f"  Position size: ${advisor.position_size:.0f}")
    print(f"  Stop loss: {advisor.stop_loss_pct*100:.1f}%")
    print(f"  Take profit: {advisor.take_profit_pct*100:.1f}%")

    # Show current stats
    stats = mongo.get_stats()
    print(f"\nCurrent Stats:")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print(f"  Total PnL: ${stats['total_pnl']:.2f}")

    if args.once:
        advisor.run_once()
        return

    # Main loop
    try:
        while True:
            advisor.run_once()
            print(f"  Sleeping {args.interval}s...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nStopping AI Advisor...")

        # Final stats
        stats = mongo.get_stats()
        print(f"\nFinal Stats:")
        print(f"  Total trades: {stats['total_trades']}")
        print(f"  Winning: {stats['winning_trades']}")
        print(f"  Losing: {stats['losing_trades']}")
        print(f"  Win rate: {stats['win_rate']:.1f}%")
        print(f"  Total PnL: ${stats['total_pnl']:.2f}")
        print(f"  Avg PnL: ${stats['avg_pnl']:.2f}")


if __name__ == '__main__':
    main()
