# Research Status - Retracement Strategy Analysis

**Last Updated**: 2025-12-18
**Data Collected**: ~31 hours of 1-min BTC/USD trades
**Status**: WAITING FOR MORE DATA

## Summary of Findings

### Hypothesis Tested
"After large price drops, BTC rebounds (retracement) and we can profit by buying the dip."

### Result: NO EDGE FOUND

After testing 26+ configurations across multiple strategies:

| Strategy | Result |
|----------|--------|
| Retracement (buy after drop) | ❌ Price doesn't rebound |
| Snapback (low TP/SL) | ❌ EV = -costs (~-0.5%) |
| Breakout post-consolidation | ❌ No upside breakouts |
| 5-min timeframe | ❌ Drift is NEGATIVE (continuation) |

### Key Diagnostic Data

**Post-shock drift (5-min bars)**:
- All drifts are NEGATIVE after drops
- Price continues falling, doesn't rebound
- Pattern = CONTINUATION, not retracement

| Lookback | Threshold | Drift 30min |
|----------|-----------|-------------|
| 5min | 0.5% | -1.17% |
| 10min | 1.2% | -1.20% |
| 15min | 1.0% | -0.60% |

### What Would Work
- **SHORT continuation** - but user cannot short on Kraken spot

## Scripts Created

1. `scripts/retracement_grid.py` - Tests 26 retracement/snapback configs
2. `scripts/breakout_grid.py` - Tests breakout post-range strategy
3. `scripts/ml_analysis.py` - ML labeling and analysis

## Data Collection

- **Recorder**: Running on Fly.io (background)
- **Location**: `data/crypto/trade/YYYY-MM-DD/*.parquet`
- **Pairs**: BTC/USD, ETH/USD, SOL/USD, and stocks

## Next Steps (When More Data Available)

1. **Re-run analysis** after 1-2 weeks:
   ```bash
   python3 scripts/retracement_grid.py
   ```

2. **Check for regime changes** - maybe different market conditions show retracement

3. **Consider alternatives**:
   - Different asset (not BTC)
   - Longer timeframe (15m, 1h)
   - Different pattern (momentum, not mean-reversion)

## Costs Assumed

- Taker: 0.4%
- Slippage: 0.05%
- Round-trip: ~0.9%

For strategies to work, need movements > 0.9% with good win rate.

## MongoDB Integration

UI now connects to MongoDB to display AI signals (when AI advisor is running).
- Events: `AiSignalUpdate`, `AiStatsUpdate`, `AiTradesUpdate`
- Polling: Every 2 seconds
