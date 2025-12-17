//! Input state management for the TUI
//!
//! Handles different input modes (normal, dialog, search) and focus areas.

use crate::trading::order::{OrderSide, OrderType};

/// Current input mode
#[derive(Debug, Clone, PartialEq)]
pub enum InputMode {
    /// Normal navigation mode
    Normal,
    /// Modal dialog is active
    Dialog(DialogType),
    /// Search/filter mode
    Search,
}

/// Type of dialog currently active
#[derive(Debug, Clone, PartialEq)]
pub enum DialogType {
    Help,
    Buy,
    Sell,
    Confirm(ConfirmAction),
}

/// Action to be confirmed
#[derive(Debug, Clone, PartialEq)]
pub enum ConfirmAction {
    PlaceOrder,
    CancelOrder(String),
    Quit,
}

/// Currently focused UI area
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocusArea {
    PairSelector,
    Chart,
    OrderBook,
    Positions,
    Trades,
}

impl FocusArea {
    /// Get the next focus area (clockwise navigation)
    pub fn next(self) -> Self {
        match self {
            FocusArea::PairSelector => FocusArea::Chart,
            FocusArea::Chart => FocusArea::OrderBook,
            FocusArea::OrderBook => FocusArea::Positions,
            FocusArea::Positions => FocusArea::Trades,
            FocusArea::Trades => FocusArea::PairSelector,
        }
    }

    /// Get the previous focus area (counter-clockwise navigation)
    pub fn prev(self) -> Self {
        match self {
            FocusArea::PairSelector => FocusArea::Trades,
            FocusArea::Chart => FocusArea::PairSelector,
            FocusArea::OrderBook => FocusArea::Chart,
            FocusArea::Positions => FocusArea::OrderBook,
            FocusArea::Trades => FocusArea::Positions,
        }
    }
}

/// Current view being displayed
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum View {
    #[default]
    Trading,
    Portfolio,
    History,
}

impl View {
    pub fn title(&self) -> &str {
        match self {
            View::Trading => "Trading",
            View::Portfolio => "Portfolio",
            View::History => "History",
        }
    }

    pub fn shortcut(&self) -> char {
        match self {
            View::Trading => '1',
            View::Portfolio => '2',
            View::History => '3',
        }
    }
}

/// Complete input state
#[derive(Debug, Clone)]
pub struct InputState {
    /// Current input mode
    pub mode: InputMode,
    /// Currently focused area
    pub focus: FocusArea,
    /// Current view
    pub view: View,
    /// Search query (when in search mode)
    pub search_query: String,
    /// Filtered pair indices (when search is active)
    pub filtered_pairs: Option<Vec<usize>>,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            mode: InputMode::Normal,
            focus: FocusArea::Chart,
            view: View::Trading,
            search_query: String::new(),
            filtered_pairs: None,
        }
    }
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if we're in normal mode
    pub fn is_normal(&self) -> bool {
        matches!(self.mode, InputMode::Normal)
    }

    /// Check if a dialog is active
    pub fn is_dialog(&self) -> bool {
        matches!(self.mode, InputMode::Dialog(_))
    }

    /// Check if search mode is active
    pub fn is_search(&self) -> bool {
        matches!(self.mode, InputMode::Search)
    }

    /// Enter search mode
    pub fn enter_search(&mut self) {
        self.mode = InputMode::Search;
        self.search_query.clear();
        self.filtered_pairs = None;
    }

    /// Exit search mode
    pub fn exit_search(&mut self) {
        self.mode = InputMode::Normal;
        self.search_query.clear();
        self.filtered_pairs = None;
    }

    /// Update search query and filter pairs
    pub fn update_search(&mut self, pairs: &[String]) {
        if self.search_query.is_empty() {
            self.filtered_pairs = None;
        } else {
            let query = self.search_query.to_uppercase();
            let filtered: Vec<usize> = pairs
                .iter()
                .enumerate()
                .filter(|(_, p)| p.to_uppercase().contains(&query))
                .map(|(i, _)| i)
                .collect();
            self.filtered_pairs = Some(filtered);
        }
    }

    /// Show help dialog
    pub fn show_help(&mut self) {
        self.mode = InputMode::Dialog(DialogType::Help);
    }

    /// Show buy dialog
    pub fn show_buy(&mut self) {
        self.mode = InputMode::Dialog(DialogType::Buy);
    }

    /// Show sell dialog
    pub fn show_sell(&mut self) {
        self.mode = InputMode::Dialog(DialogType::Sell);
    }

    /// Show confirm dialog
    pub fn show_confirm(&mut self, action: ConfirmAction) {
        self.mode = InputMode::Dialog(DialogType::Confirm(action));
    }

    /// Close current dialog
    pub fn close_dialog(&mut self) {
        self.mode = InputMode::Normal;
    }

    /// Set view
    pub fn set_view(&mut self, view: View) {
        self.view = view;
    }

    /// Navigate focus to next area
    pub fn focus_next(&mut self) {
        self.focus = self.focus.next();
    }

    /// Navigate focus to previous area
    pub fn focus_prev(&mut self) {
        self.focus = self.focus.prev();
    }
}

/// State for the order entry dialog
#[derive(Debug, Clone)]
pub struct OrderDialogState {
    pub pair: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: String,
    pub price: String,
    pub focused_field: OrderDialogField,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderDialogField {
    OrderType,
    Quantity,
    Price,
    Submit,
    Cancel,
}

impl OrderDialogField {
    pub fn next(self) -> Self {
        match self {
            OrderDialogField::OrderType => OrderDialogField::Quantity,
            OrderDialogField::Quantity => OrderDialogField::Price,
            OrderDialogField::Price => OrderDialogField::Submit,
            OrderDialogField::Submit => OrderDialogField::Cancel,
            OrderDialogField::Cancel => OrderDialogField::OrderType,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            OrderDialogField::OrderType => OrderDialogField::Cancel,
            OrderDialogField::Quantity => OrderDialogField::OrderType,
            OrderDialogField::Price => OrderDialogField::Quantity,
            OrderDialogField::Submit => OrderDialogField::Price,
            OrderDialogField::Cancel => OrderDialogField::Submit,
        }
    }
}

impl OrderDialogState {
    pub fn new_buy(pair: &str) -> Self {
        Self {
            pair: pair.to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: String::new(),
            price: String::new(),
            focused_field: OrderDialogField::Quantity,
            error_message: None,
        }
    }

    pub fn new_sell(pair: &str) -> Self {
        Self {
            pair: pair.to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Market,
            quantity: String::new(),
            price: String::new(),
            focused_field: OrderDialogField::Quantity,
            error_message: None,
        }
    }

    pub fn toggle_order_type(&mut self) {
        self.order_type = match self.order_type {
            OrderType::Market => OrderType::Limit,
            OrderType::Limit => OrderType::Market,
            // StopLoss and TakeProfit are not used in the dialog
            OrderType::StopLoss | OrderType::TakeProfit => OrderType::Market,
        };
    }

    pub fn focus_next(&mut self) {
        self.focused_field = self.focused_field.next();
    }

    pub fn focus_prev(&mut self) {
        self.focused_field = self.focused_field.prev();
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    pub fn set_error(&mut self, msg: &str) {
        self.error_message = Some(msg.to_string());
    }
}
