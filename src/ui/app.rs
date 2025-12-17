//! Main TUI application

use crate::api::{KrakenWebSocket, MarketEvent};
use crate::config::Config;
use crate::data::{Candle, OrderBook, Ticker};
use crate::storage::{DataRecorder, Database, TradeRecord};
use crate::trading::order::Position;
use crate::trading::TradingEngine;
use crate::ui::dialogs::{ConfirmDialog, HelpOverlay, OrderDialog};
use crate::ui::input::{
    ConfirmAction, DialogType, InputMode, InputState, OrderDialogField, OrderDialogState, View,
};
use crate::ui::views::{HistoryView, PortfolioView, TradingView, ViewRenderer, ViewState};
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::error;

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
    // New fields for enhanced UI
    input: InputState,
    order_dialog: Option<OrderDialogState>,
    confirm_dialog: Option<(String, String, ConfirmAction, bool)>, // (title, message, action, selected_yes)
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
            input: InputState::new(),
            order_dialog: None,
            confirm_dialog: None,
        }
    }
}

pub struct App {
    config: Config,
    engine: Arc<Mutex<TradingEngine>>,
    recorder: DataRecorder,
    recent_trades: Vec<TradeRecord>,
    connected: bool,
    selected_pair: usize,
    should_quit: bool,
    ui_state: UiState,
    all_pairs: Vec<String>,
}

impl App {
    pub async fn new(config: Config) -> Result<Self> {
        let db = Arc::new(Database::new(&config.database.path).await?);
        let engine = TradingEngine::new(config.clone(), Arc::clone(&db)).await?;

        let recent_trades = db.get_recent_trades(50).await.unwrap_or_default();

        let ui_state = UiState::new(engine.balance(), engine.is_paper_trading());

        // Create data recorder
        let recorder = DataRecorder::new(&config);

        let all_pairs = config.all_pairs();

        Ok(Self {
            config,
            engine: Arc::new(Mutex::new(engine)),
            recorder,
            recent_trades,
            connected: false,
            selected_pair: 0,
            should_quit: false,
            ui_state,
            all_pairs,
        })
    }

    fn selected_pair(&self) -> &str {
        self.all_pairs
            .get(self.selected_pair)
            .map(|s| s.as_str())
            .unwrap_or("BTC/USD")
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

        // Start WebSocket connection with separate crypto/stock pairs and depths
        let ws = KrakenWebSocket::new(
            self.config.kraken.clone(),
            self.config.trading.crypto_pairs.clone(),
            self.config.trading.stock_pairs.clone(),
            self.config.recording.crypto_book_depth,
            self.config.recording.stock_book_depth,
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

            // Sample data for recording if enabled
            if self.recorder.is_enabled() {
                self.sample_data_for_recording().await;

                // Flush to disk if it's time
                if self.recorder.should_flush() {
                    if let Err(e) = self.recorder.flush() {
                        error!("Failed to flush data: {}", e);
                    }
                }
            }

            // Draw UI
            terminal.draw(|f| self.draw(f))?;

            // Handle input
            if event::poll(tick_rate)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key_input(key.code).await;
                    }
                }
            }

            if self.should_quit {
                break;
            }
        }

        // Flush any remaining data before exit
        if self.recorder.is_enabled() {
            if let Err(e) = self.recorder.flush() {
                error!("Failed to flush data on exit: {}", e);
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

    /// Handle keyboard input based on current mode
    async fn handle_key_input(&mut self, key: KeyCode) {
        match &self.ui_state.input.mode {
            InputMode::Normal => self.handle_normal_input(key).await,
            InputMode::Dialog(dialog_type) => {
                self.handle_dialog_input(key, dialog_type.clone()).await
            }
            InputMode::Search => self.handle_search_input(key),
        }
    }

    /// Handle input in normal mode
    async fn handle_normal_input(&mut self, key: KeyCode) {
        match key {
            // Quit
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }

            // View switching
            KeyCode::Char('1') => {
                self.ui_state.input.set_view(View::Trading);
            }
            KeyCode::Char('2') => {
                self.ui_state.input.set_view(View::Portfolio);
            }
            KeyCode::Char('3') => {
                self.ui_state.input.set_view(View::History);
            }

            // Pair selection
            KeyCode::Tab => {
                self.selected_pair = (self.selected_pair + 1) % self.all_pairs.len();
            }
            KeyCode::BackTab => {
                if self.selected_pair == 0 {
                    self.selected_pair = self.all_pairs.len().saturating_sub(1);
                } else {
                    self.selected_pair -= 1;
                }
            }

            // Navigation
            KeyCode::Up => {
                self.ui_state.input.focus_prev();
            }
            KeyCode::Down => {
                self.ui_state.input.focus_next();
            }

            // Help
            KeyCode::Char('?') | KeyCode::Char('h') | KeyCode::Char('H') => {
                self.ui_state.input.show_help();
            }

            // Search
            KeyCode::Char('/') => {
                self.ui_state.input.enter_search();
            }

            // Trading dialogs
            KeyCode::Char('b') | KeyCode::Char('B') => {
                let pair = self.selected_pair();
                self.ui_state.order_dialog = Some(OrderDialogState::new_buy(pair));
                self.ui_state.input.show_buy();
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                let pair = self.selected_pair();
                self.ui_state.order_dialog = Some(OrderDialogState::new_sell(pair));
                self.ui_state.input.show_sell();
            }

            _ => {}
        }
    }

    /// Handle input in dialog mode
    async fn handle_dialog_input(&mut self, key: KeyCode, dialog_type: DialogType) {
        match dialog_type {
            DialogType::Help => {
                // Any key closes help
                self.ui_state.input.close_dialog();
            }
            DialogType::Buy | DialogType::Sell => {
                self.handle_order_dialog_input(key).await;
            }
            DialogType::Confirm(_) => {
                self.handle_confirm_dialog_input(key).await;
            }
        }
    }

    /// Handle input in order dialog
    async fn handle_order_dialog_input(&mut self, key: KeyCode) {
        let Some(ref mut dialog) = self.ui_state.order_dialog else {
            return;
        };

        match key {
            KeyCode::Esc => {
                self.ui_state.order_dialog = None;
                self.ui_state.input.close_dialog();
            }
            KeyCode::Tab | KeyCode::Down => {
                dialog.focus_next();
            }
            KeyCode::BackTab | KeyCode::Up => {
                dialog.focus_prev();
            }
            KeyCode::Enter => {
                match dialog.focused_field {
                    OrderDialogField::Cancel => {
                        self.ui_state.order_dialog = None;
                        self.ui_state.input.close_dialog();
                    }
                    OrderDialogField::Submit => {
                        // Validate and submit order
                        if let Err(e) = self.validate_and_submit_order().await {
                            if let Some(ref mut d) = self.ui_state.order_dialog {
                                d.set_error(&e);
                            }
                        }
                    }
                    OrderDialogField::OrderType => {
                        dialog.toggle_order_type();
                    }
                    _ => {}
                }
            }
            KeyCode::Left | KeyCode::Right => {
                if dialog.focused_field == OrderDialogField::OrderType {
                    dialog.toggle_order_type();
                }
            }
            // Quantity shortcuts
            KeyCode::Char('1') if dialog.focused_field == OrderDialogField::Quantity => {
                self.set_quantity_percent(25);
            }
            KeyCode::Char('2') if dialog.focused_field == OrderDialogField::Quantity => {
                self.set_quantity_percent(50);
            }
            KeyCode::Char('3') if dialog.focused_field == OrderDialogField::Quantity => {
                self.set_quantity_percent(75);
            }
            KeyCode::Char('4') if dialog.focused_field == OrderDialogField::Quantity => {
                self.set_quantity_percent(100);
            }
            // Text input
            KeyCode::Char(c) => {
                if c.is_ascii_digit() || c == '.' {
                    match dialog.focused_field {
                        OrderDialogField::Quantity => {
                            dialog.quantity.push(c);
                            dialog.clear_error();
                        }
                        OrderDialogField::Price => {
                            dialog.price.push(c);
                            dialog.clear_error();
                        }
                        _ => {}
                    }
                }
            }
            KeyCode::Backspace => {
                match dialog.focused_field {
                    OrderDialogField::Quantity => {
                        dialog.quantity.pop();
                    }
                    OrderDialogField::Price => {
                        dialog.price.pop();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn set_quantity_percent(&mut self, percent: u32) {
        let Some(ref mut dialog) = self.ui_state.order_dialog else {
            return;
        };

        let current_price = self
            .ui_state
            .tickers
            .get(&dialog.pair)
            .map(|t| t.last)
            .unwrap_or(Decimal::ZERO);

        if current_price > Decimal::ZERO {
            let available = self.ui_state.balance * Decimal::from(percent) / Decimal::from(100);
            let qty = available / current_price;
            dialog.quantity = format!("{:.6}", qty);
        }
    }

    async fn validate_and_submit_order(&mut self) -> Result<(), String> {
        let Some(ref dialog) = self.ui_state.order_dialog else {
            return Err("No order dialog".into());
        };

        // Parse quantity
        let quantity: Decimal = dialog
            .quantity
            .parse()
            .map_err(|_| "Invalid quantity")?;

        if quantity <= Decimal::ZERO {
            return Err("Quantity must be positive".into());
        }

        // Get current price
        let current_price = self
            .ui_state
            .tickers
            .get(&dialog.pair)
            .map(|t| t.last)
            .unwrap_or(Decimal::ZERO);

        // Parse price for limit orders
        let price = if dialog.order_type == crate::trading::order::OrderType::Limit {
            let p: Decimal = dialog.price.parse().map_err(|_| "Invalid price")?;
            if p <= Decimal::ZERO {
                return Err("Price must be positive".into());
            }
            Some(p)
        } else {
            None
        };

        // Check balance for buys
        if dialog.side == crate::trading::order::OrderSide::Buy {
            let cost = quantity * price.unwrap_or(current_price);
            if cost > self.ui_state.balance {
                return Err(format!(
                    "Insufficient balance. Need ${:.2}, have ${:.2}",
                    cost, self.ui_state.balance
                ));
            }
        }

        // Create order using the factory methods
        let order = if let Some(limit_price) = price {
            crate::trading::order::Order::limit(&dialog.pair, dialog.side.clone(), quantity, limit_price)
        } else {
            crate::trading::order::Order::market(&dialog.pair, dialog.side.clone(), quantity)
        };

        // Execute order
        let ticker = self
            .ui_state
            .tickers
            .get(&dialog.pair)
            .cloned()
            .ok_or("No ticker data for pair")?;

        let mut engine = self.engine.lock().await;
        match engine.execute_order(order, &ticker).await {
            Ok(_executed) => {
                // Success - close dialog
                drop(engine);
                self.ui_state.order_dialog = None;
                self.ui_state.input.close_dialog();
                Ok(())
            }
            Err(e) => Err(format!("Order failed: {}", e)),
        }
    }

    /// Handle input in confirm dialog
    async fn handle_confirm_dialog_input(&mut self, key: KeyCode) {
        let Some((_, _, ref action, ref mut selected_yes)) = self.ui_state.confirm_dialog else {
            return;
        };

        match key {
            KeyCode::Esc => {
                self.ui_state.confirm_dialog = None;
                self.ui_state.input.close_dialog();
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Tab => {
                *selected_yes = !*selected_yes;
            }
            KeyCode::Enter => {
                if *selected_yes {
                    // Execute action
                    match action {
                        ConfirmAction::Quit => {
                            self.should_quit = true;
                        }
                        ConfirmAction::PlaceOrder => {
                            // Order placement handled elsewhere
                        }
                        ConfirmAction::CancelOrder(_txid) => {
                            // Cancel order
                        }
                    }
                }
                self.ui_state.confirm_dialog = None;
                self.ui_state.input.close_dialog();
            }
            _ => {}
        }
    }

    /// Handle input in search mode
    fn handle_search_input(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => {
                self.ui_state.input.exit_search();
            }
            KeyCode::Enter => {
                // Select first filtered pair if any
                if let Some(ref filtered) = self.ui_state.input.filtered_pairs {
                    if let Some(&idx) = filtered.first() {
                        self.selected_pair = idx;
                    }
                }
                self.ui_state.input.exit_search();
            }
            KeyCode::Char(c) => {
                self.ui_state.input.search_query.push(c);
                self.ui_state.input.update_search(&self.all_pairs);
            }
            KeyCode::Backspace => {
                self.ui_state.input.search_query.pop();
                self.ui_state.input.update_search(&self.all_pairs);
            }
            _ => {}
        }
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
            for pair in &self.all_pairs {
                if let Some(ob) = orderbooks.get(pair) {
                    self.ui_state.orderbooks.insert(pair.clone(), ob.clone());
                }
            }
        }

        // Sync candles
        if let Ok(candles) = ws_candles.try_lock() {
            for pair in &self.all_pairs {
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

    /// Sample data for recording based on configured intervals
    async fn sample_data_for_recording(&mut self) {
        // Sample crypto data
        if self.recorder.should_sample_crypto() {
            for pair in &self.config.trading.crypto_pairs {
                // Sample ticker
                if let Some(ticker) = self.ui_state.tickers.get(pair) {
                    self.recorder.record_ticker(ticker);
                }

                // Sample orderbook
                if let Some(orderbook) = self.ui_state.orderbooks.get(pair) {
                    self.recorder.record_orderbook(pair, orderbook);
                }

                // Sample latest candle
                if let Some(candles) = self.ui_state.candles.get(pair) {
                    if let Some(candle) = candles.last() {
                        self.recorder.record_ohlc(pair, candle);
                    }
                }
            }
            self.recorder.mark_crypto_sampled();
        }

        // Sample stock data
        if self.recorder.should_sample_stock() {
            for pair in &self.config.trading.stock_pairs {
                // Sample ticker
                if let Some(ticker) = self.ui_state.tickers.get(pair) {
                    self.recorder.record_ticker(ticker);
                }

                // Sample orderbook
                if let Some(orderbook) = self.ui_state.orderbooks.get(pair) {
                    self.recorder.record_orderbook(pair, orderbook);
                }

                // Sample latest candle
                if let Some(candles) = self.ui_state.candles.get(pair) {
                    if let Some(candle) = candles.last() {
                        self.recorder.record_ohlc(pair, candle);
                    }
                }
            }
            self.recorder.mark_stock_sampled();
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
            MarketEvent::TradeUpdate(trade) => {
                // Record every trade in real-time (no sampling for trades)
                if self.recorder.is_enabled() {
                    self.recorder.record_trade(
                        &trade.pair,
                        &trade.side,
                        trade.price,
                        trade.qty,
                        trade.trade_id,
                    );
                }
            }
            MarketEvent::Error(err) => {
                tracing::error!("Market error: {}", err);
            }
        }
    }

    fn draw(&self, f: &mut ratatui::Frame) {
        let size = f.area();

        // Create view state
        let view_state = ViewState {
            tickers: &self.ui_state.tickers,
            orderbooks: &self.ui_state.orderbooks,
            candles: &self.ui_state.candles,
            positions: &self.ui_state.positions,
            recent_trades: &self.recent_trades,
            balance: self.ui_state.balance,
            equity: self.ui_state.equity,
            total_pnl: self.ui_state.total_pnl,
            total_pnl_pct: self.ui_state.total_pnl_pct,
            is_paper: self.ui_state.is_paper,
            connected: self.connected,
            selected_pair: self.selected_pair(),
            all_pairs: &self.all_pairs,
        };

        // Render current view
        match self.ui_state.input.view {
            View::Trading => TradingView::new().render(f, size, &view_state),
            View::Portfolio => PortfolioView::new().render(f, size, &view_state),
            View::History => HistoryView::new().render(f, size, &view_state),
        }

        // Render overlay dialogs
        self.draw_dialogs(f, size);
    }

    fn draw_dialogs(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        match &self.ui_state.input.mode {
            InputMode::Dialog(DialogType::Help) => {
                f.render_widget(HelpOverlay::new(), area);
            }
            InputMode::Dialog(DialogType::Buy) | InputMode::Dialog(DialogType::Sell) => {
                if let Some(ref dialog_state) = self.ui_state.order_dialog {
                    let current_price = self
                        .ui_state
                        .tickers
                        .get(&dialog_state.pair)
                        .map(|t| t.last);
                    let widget =
                        OrderDialog::new(dialog_state, current_price, self.ui_state.balance);
                    f.render_widget(widget, area);
                }
            }
            InputMode::Dialog(DialogType::Confirm(_)) => {
                if let Some((ref title, ref msg, _, selected_yes)) = self.ui_state.confirm_dialog {
                    let widget = ConfirmDialog::new(title, msg, selected_yes);
                    f.render_widget(widget, area);
                }
            }
            InputMode::Search => {
                // Draw search overlay at top
                self.draw_search_overlay(f, area);
            }
            InputMode::Normal => {}
        }
    }

    fn draw_search_overlay(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        use ratatui::{
            layout::Rect,
            style::{Color, Style},
            widgets::{Block, Borders, Clear, Paragraph, Widget},
        };

        let search_area = Rect {
            x: area.width / 4,
            y: 2,
            width: area.width / 2,
            height: 3,
        };

        Clear.render(search_area, f.buffer_mut());

        let block = Block::default()
            .title(" Search Pairs (Enter to select, Esc to cancel) ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .style(Style::default().bg(Color::Black));

        let query_display = format!("> {}_", self.ui_state.input.search_query);
        let paragraph = Paragraph::new(query_display)
            .block(block)
            .style(Style::default().fg(Color::White));

        f.render_widget(paragraph, search_area);

        // Show filtered results below
        if let Some(ref filtered) = self.ui_state.input.filtered_pairs {
            let results_area = Rect {
                x: search_area.x,
                y: search_area.y + 3,
                width: search_area.width,
                height: (filtered.len() as u16).min(5) + 2,
            };

            if !filtered.is_empty() {
                Clear.render(results_area, f.buffer_mut());

                let results: Vec<String> = filtered
                    .iter()
                    .take(5)
                    .filter_map(|&i| self.all_pairs.get(i).cloned())
                    .collect();

                let block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .style(Style::default().bg(Color::Black));

                let paragraph = Paragraph::new(results.join("\n"))
                    .block(block)
                    .style(Style::default().fg(Color::White));

                f.render_widget(paragraph, results_area);
            }
        }
    }
}
