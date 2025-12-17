use crate::api::{websocket::MarketEvent, KrakenWebSocket};
use crate::config::Config;
use crate::data::{Candle, OrderBook, Ticker};
use crate::storage::{Database, TradeRecord};
use crate::trading::order::Position;
use crate::trading::TradingEngine;
use crate::ui::charts::PriceChart;
use crate::ui::widgets::{
    orderbook::OrderBookWidget, positions::PositionsWidget, status::StatusWidget,
    trades::TradesWidget,
};
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Terminal,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};

/// Local UI state - no locks needed for rendering
struct UiState {
    tickers: HashMap<String, Ticker>,
    orderbooks: HashMap<String, OrderBook>,
    candles: HashMap<String, Vec<Candle>>,
    positions: Vec<Position>,
    balance: Decimal,
    equity: Decimal,
    total_pnl: Decimal,
    total_pnl_pct: Decimal,
    is_paper: bool,
}

impl UiState {
    fn new(balance: Decimal, is_paper: bool) -> Self {
        Self {
            tickers: HashMap::new(),
            orderbooks: HashMap::new(),
            candles: HashMap::new(),
            positions: Vec::new(),
            balance,
            equity: balance,
            total_pnl: Decimal::ZERO,
            total_pnl_pct: Decimal::ZERO,
            is_paper,
        }
    }
}

pub struct App {
    config: Config,
    engine: Arc<Mutex<TradingEngine>>,
    recent_trades: Vec<TradeRecord>,
    connected: bool,
    selected_pair: usize,
    should_quit: bool,
    ui_state: UiState,
}

impl App {
    pub async fn new(config: Config) -> Result<Self> {
        let db = Arc::new(Database::new(&config.database.path).await?);
        let engine = TradingEngine::new(config.clone(), Arc::clone(&db)).await?;

        let recent_trades = db.get_recent_trades(20).await.unwrap_or_default();

        let ui_state = UiState::new(engine.balance(), engine.is_paper_trading());

        Ok(Self {
            config,
            engine: Arc::new(Mutex::new(engine)),
            recent_trades,
            connected: false,
            selected_pair: 0,
            should_quit: false,
            ui_state,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Create event channel
        let (event_tx, mut event_rx) = mpsc::channel::<MarketEvent>(100);

        // Start WebSocket connection
        let ws = KrakenWebSocket::new(
            self.config.kraken.clone(),
            self.config.trading.pairs.clone(),
            self.config.ui.chart_candles,
        );

        let ws_tickers = Arc::clone(&ws.tickers);
        let ws_orderbooks = Arc::clone(&ws.orderbooks);
        let ws_candles = Arc::clone(&ws.candles);

        // Spawn WebSocket task
        let ws_event_tx = event_tx.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = ws.connect(ws_event_tx.clone()).await {
                    tracing::error!("WebSocket error: {}", e);
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
                tracing::info!("Reconnecting to WebSocket...");
            }
        });

        // Main loop
        let tick_rate = Duration::from_millis(self.config.ui.refresh_rate_ms);

        loop {
            // Handle market events
            while let Ok(event) = event_rx.try_recv() {
                self.handle_market_event(event).await;
            }

            // Sync UI state from shared data (using try_lock to avoid blocking)
            self.sync_ui_state(&ws_tickers, &ws_orderbooks, &ws_candles)
                .await;

            // Draw UI
            terminal.draw(|f| self.draw(f))?;

            // Handle input
            if event::poll(tick_rate)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                self.should_quit = true;
                            }
                            KeyCode::Tab => {
                                self.selected_pair =
                                    (self.selected_pair + 1) % self.config.trading.pairs.len();
                            }
                            _ => {}
                        }
                    }
                }
            }

            if self.should_quit {
                break;
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(())
    }

    async fn sync_ui_state(
        &mut self,
        ws_tickers: &Arc<tokio::sync::Mutex<crate::data::TickerStore>>,
        ws_orderbooks: &Arc<tokio::sync::Mutex<crate::data::OrderBookStore>>,
        ws_candles: &Arc<tokio::sync::Mutex<crate::data::CandleStore>>,
    ) {
        // Sync tickers
        if let Ok(tickers) = ws_tickers.try_lock() {
            for ticker in tickers.all() {
                self.ui_state
                    .tickers
                    .insert(ticker.pair.clone(), ticker.clone());
            }
        }

        // Sync orderbooks
        if let Ok(orderbooks) = ws_orderbooks.try_lock() {
            for pair in &self.config.trading.pairs {
                if let Some(ob) = orderbooks.get(pair) {
                    self.ui_state.orderbooks.insert(pair.clone(), ob.clone());
                }
            }
        }

        // Sync candles
        if let Ok(candles) = ws_candles.try_lock() {
            for pair in &self.config.trading.pairs {
                if let Some(pair_candles) = candles.get(pair) {
                    self.ui_state
                        .candles
                        .insert(pair.clone(), pair_candles.iter().cloned().collect());
                }
            }
        }

        // Sync engine state
        if let Ok(engine) = self.engine.try_lock() {
            self.ui_state.balance = engine.balance();
            self.ui_state.equity = engine.equity();
            self.ui_state.total_pnl = engine.total_pnl();
            self.ui_state.total_pnl_pct = engine.total_pnl_pct();
            self.ui_state.positions = engine.positions().into_iter().cloned().collect();
        }
    }

    async fn handle_market_event(&mut self, event: MarketEvent) {
        match event {
            MarketEvent::Connected => {
                self.connected = true;
                tracing::info!("Connected to Kraken WebSocket");
            }
            MarketEvent::Disconnected => {
                self.connected = false;
                tracing::info!("Disconnected from Kraken WebSocket");
            }
            MarketEvent::TickerUpdate(ticker) => {
                // Update positions with new price
                if let Ok(mut engine) = self.engine.try_lock() {
                    engine.update_positions(&ticker);
                }
            }
            MarketEvent::OrderBookUpdate(_pair) => {}
            MarketEvent::CandleUpdate(_pair) => {}
            MarketEvent::Error(err) => {
                tracing::error!("Market error: {}", err);
            }
        }
    }

    fn draw(&self, f: &mut ratatui::Frame) {
        let size = f.area();

        // Main layout: status bar + content
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(10)])
            .split(size);

        // Render status bar
        self.draw_status(f, main_chunks[0]);

        // Content layout: left (chart + orderbook) + right (positions + trades)
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(main_chunks[1]);

        // Left side: chart on top, orderbook on bottom
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(content_chunks[0]);

        self.draw_chart(f, left_chunks[0]);
        self.draw_orderbook(f, left_chunks[1]);

        // Right side: positions on top, trades on bottom
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content_chunks[1]);

        self.draw_positions(f, right_chunks[0]);
        self.draw_trades(f, right_chunks[1]);
    }

    fn draw_status(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let tickers: Vec<&Ticker> = self.ui_state.tickers.values().collect();

        let widget = StatusWidget::new(
            tickers,
            self.ui_state.balance,
            self.ui_state.equity,
            self.ui_state.total_pnl,
            self.ui_state.total_pnl_pct,
            self.ui_state.is_paper,
            self.connected,
        );

        f.render_widget(widget, area);
    }

    fn draw_chart(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let pair = self
            .config
            .trading
            .pairs
            .get(self.selected_pair)
            .map(|s| s.as_str())
            .unwrap_or("BTC/USD");

        let candles: Vec<&Candle> = self
            .ui_state
            .candles
            .get(pair)
            .map(|c| c.iter().collect())
            .unwrap_or_default();

        let widget = PriceChart::new(&candles, pair);
        f.render_widget(widget, area);
    }

    fn draw_orderbook(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let pair = self
            .config
            .trading
            .pairs
            .get(self.selected_pair)
            .map(|s| s.as_str())
            .unwrap_or("BTC/USD");

        let orderbook = self.ui_state.orderbooks.get(pair);

        let widget = OrderBookWidget::new(orderbook).depth(10);
        f.render_widget(widget, area);
    }

    fn draw_positions(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let positions: Vec<&Position> = self.ui_state.positions.iter().collect();
        let widget = PositionsWidget::new(positions);
        f.render_widget(widget, area);
    }

    fn draw_trades(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let widget = TradesWidget::new(&self.recent_trades);
        f.render_widget(widget, area);
    }
}
