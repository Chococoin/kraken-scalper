pub mod base;
pub mod breakout;
pub mod indicators;
pub mod mean_reversion;
pub mod registry;
pub mod rsi;
pub mod scalper;
pub mod trend_following;

pub use base::{Signal, Strategy};
pub use breakout::{BreakoutConfig, BreakoutStrategy};
pub use mean_reversion::{MeanReversionConfig, MeanReversionStrategy};
pub use registry::{all_strategies, create_strategy, parse_strategy_list, STRATEGY_NAMES};
pub use rsi::{RSIConfig, RSIStrategy};
pub use scalper::ScalperStrategy;
pub use trend_following::{TrendFollowingConfig, TrendFollowingStrategy};
