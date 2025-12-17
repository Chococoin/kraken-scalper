#!/usr/bin/env python3
"""
Initialize HuggingFace dataset repository.
Run once to create the dataset and upload initial README.

Usage:
    HF_TOKEN=xxx python scripts/hf_init.py username/kraken-market-data

Requirements:
    pip install huggingface_hub
"""

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Run: pip install huggingface_hub")
    sys.exit(1)

DATASET_CARD = """---
language:
- en
license: mit
task_categories:
- time-series-forecasting
tags:
- finance
- cryptocurrency
- trading
- market-data
- parquet
- kraken
pretty_name: Kraken Market Data
size_categories:
- 1K<n<10K
---

# Kraken Market Data

Real-time market data recorded from Kraken exchange WebSocket API.

## Dataset Structure

```
data/
  crypto/
    ticker/      # Price snapshots (bid, ask, last, volume, etc.)
    book/        # Order book depth (bids/asks as JSON)
    ohlc/        # OHLC candlestick data
    trade/       # Individual trade executions
```

## Data Types

### Ticker Schema
| Column | Type | Description |
|--------|------|-------------|
| ts | Int64 | Timestamp (milliseconds since epoch) |
| pair | String | Trading pair (e.g., "BTC/USD") |
| bid | Float64 | Best bid price |
| ask | Float64 | Best ask price |
| last | Float64 | Last trade price |
| volume | Float64 | 24h volume |
| vwap | Float64 | 24h VWAP |
| high | Float64 | 24h high |
| low | Float64 | 24h low |
| change_pct | Float64 | 24h change percentage |

### Order Book Schema
| Column | Type | Description |
|--------|------|-------------|
| ts | Int64 | Timestamp (milliseconds since epoch) |
| pair | String | Trading pair |
| bids_json | String | JSON array of [price, qty] tuples |
| asks_json | String | JSON array of [price, qty] tuples |

### OHLC Schema
| Column | Type | Description |
|--------|------|-------------|
| ts | Int64 | Candle timestamp (milliseconds) |
| pair | String | Trading pair |
| open | Float64 | Open price |
| high | Float64 | High price |
| low | Float64 | Low price |
| close | Float64 | Close price |
| volume | Float64 | Volume |
| vwap | Float64 | Volume-weighted average price |
| trades | Int64 | Number of trades |

### Trade Schema
| Column | Type | Description |
|--------|------|-------------|
| ts | Int64 | Trade timestamp (milliseconds) |
| pair | String | Trading pair |
| side | String | "buy" or "sell" |
| price | Float64 | Trade price |
| qty | Float64 | Trade quantity |
| trade_id | Int64 | Exchange trade ID |

## Usage

### Python (datasets library)
```python
from datasets import load_dataset

# Load ticker data
ds = load_dataset("{repo_id}", data_dir="data/crypto/ticker")

# Load specific date
ds = load_dataset(
    "{repo_id}",
    data_files="data/crypto/ticker/2025-12-17/*.parquet"
)
```

### Python (pandas)
```python
import pandas as pd

# Direct parquet read
df = pd.read_parquet("hf://datasets/{repo_id}/data/crypto/ticker/2025-12-17/13.parquet")
```

### Rust
```rust
use parquet::file::reader::FileReader;
// Download files from HuggingFace Hub first
```

## Pairs Tracked

**Crypto:**
- BTC/USD, BTC/EUR
- ETH/USD, ETH/EUR
- SOL/USD, SOL/EUR
- DOT/USD, DOT/EUR
- MATIC/USD, MATIC/EUR

## Data Recording

Data is recorded continuously using a Rust trading bot:
- **Crypto sampling**: Every 30 seconds
- **Flush interval**: Every hour
- **Upload delay**: 1 hour (ensures files are complete)

## License

MIT
"""


def main():
    parser = argparse.ArgumentParser(
        description="Initialize HuggingFace dataset repository"
    )
    parser.add_argument(
        "repo_id",
        help="Repository ID (username/dataset-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset"
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable required")
        print("Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi()

    # Create repository
    print(f"Creating dataset repository: {args.repo_id}")
    try:
        create_repo(
            args.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
            token=token
        )
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)

    # Upload README
    print("Uploading dataset card (README.md)...")
    readme_content = DATASET_CARD.replace("{repo_id}", args.repo_id)

    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
        )
    except Exception as e:
        print(f"Error uploading README: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Dataset initialized successfully!")
    print("=" * 60)
    print()
    print(f"URL: https://huggingface.co/datasets/{args.repo_id}")
    print()
    print("Next steps:")
    print(f"1. Update config/default.toml:")
    print(f"   [huggingface]")
    print(f"   enabled = true")
    print(f"   repo_id = \"{args.repo_id}\"")
    print()
    print("2. Run the recorder with HF_TOKEN:")
    print(f"   HF_TOKEN=\"{token[:10]}...\" cargo run --bin recorder --release")
    print()


if __name__ == "__main__":
    main()
