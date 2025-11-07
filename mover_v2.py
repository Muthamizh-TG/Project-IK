'''
this is mover_v2.py
uses REST API polling for real-time data and trade updates
fixes websocket 12 hours old data issue for crypto
but need to handle buy filling timeouts
'''
import os
import time
import logging
import argparse
import json
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
from queue import Queue


import numpy as np
import pandas as pd
import websocket

# Alpaca SDK - Using BOTH libraries:
# 1. alpaca-py for trading operations
# 2. alpaca_trade_api for real-time data (works better with paper trading)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OrderSide, TimeInForce, OrderType
from alpaca.data.live import CryptoDataStream

# Use older alpaca_trade_api for data fetching (works with paper trading)
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame as TradeApiTimeFrame

# -----------------------
# Config / constants
# -----------------------
API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
API_SECRET = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"
PAPER = True  # use paper trading for safety; change to False only when ready
SYMBOL = "XRP/USD"        # default trading pair on Alpaca crypto exchange (use format with slash)
BAR_TIMEFRAME = TradeApiTimeFrame.Minute  # use 1-minute bars for indicators (can move to 1s/tick later)
LOOKBACK_BARS = 100      # how many historical bars to fetch for indicators
TARGET_USD_PER_TRADE = 0.40   # per-trade dollar target (from earlier example)
TRADES_PER_10MIN = 6      # N_target in your plan
GROSS_PCT = 0.008         # target gross pct per trade (0.8% - must be > fees+slippage for profitability)
MAX_ACCOUNT_RISK_PCT = 0.02  # cap position size to 2% of account equity
SESSION_MAX_LOSS_PCT = 0.02   # stop trading for rest of session if loss exceeds this

# -----------------------
# BUY CONDITION TOGGLES
# Enable/disable individual buy conditions for testing
# -----------------------
BUY_CONDITIONS = {
    'UPTREND': True,      # EMA9 > EMA21 and Price > VWAP
    'GOOD_ENTRY': True,   # Price pullback within 0.5% of EMA9
    'RSI_CHECK': False,    # RSI between 40-65
}


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------
# Utility: indicators
# -----------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price
    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD line, signal line, and histogram
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# -----------------------
# Money math & sizing
# -----------------------
def estimate_fees_and_slippage() -> float:
    """
    Return round-trip fees+slippage as fraction (e.g., 0.002 == 0.2%).
    IMPORTANT: replace with measured values from your account / pair.
    For Alpaca crypto, there are crypto fees; also slippage depends on size and liquidity.
    
    Alpaca Crypto Trading Fees:
    - Maker: 0.15% per trade
    - Taker: 0.25% per trade
    - Round-trip (buy + sell): ~0.3-0.5%
    - Slippage: ~0.05-0.1% for small orders
    """
    # More realistic estimate: 0.15% maker fee + 0.05% slippage = 0.2% per side
    # Round-trip: 0.4% total (0.002 + 0.002)
    fee_pct = 0.0015          # 0.15% maker fee per trade
    round_trip_slippage_pct = 0.0005  # 0.05% slippage per trade
    return (fee_pct + round_trip_slippage_pct) * 2  # Both sides: buy + sell = 0.004 total

def compute_position_size_usd(target_usd: float, gross_pct: float, fees_and_slippage: float, account_equity: float) -> float:
    """
    Compute USD size S = P / (G - F). Cap at MAX_ACCOUNT_RISK_PCT of equity.
    Returns USD position size (not quantity).
    """
    net_pct = gross_pct - fees_and_slippage
    if net_pct <= 0:
        raise ValueError(f"net_pct <= 0 (gross={gross_pct}, fees+slippage={fees_and_slippage}); strategy not viable")
    S = target_usd / net_pct
    cap = MAX_ACCOUNT_RISK_PCT * account_equity
    return min(S, cap)

# -----------------------
# Performance Tracker
# -----------------------
class PerformanceTracker:
    """Track trading performance metrics"""
    
    def __init__(self):
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.total_slippage = 0.0
        self.fill_successes = 0
        self.fill_timeouts = 0
        
    def record_trade(self, pnl: float, entry_price: float, exit_price: float, expected_entry: float, qty: float):
        """Record a completed trade"""
        # Calculate slippage
        slippage = abs(entry_price - expected_entry) / expected_entry if expected_entry > 0 else 0.0
        
        trade_record = {
            'pnl': pnl,
            'pnl_pct': pnl / (entry_price * qty) if qty > 0 else 0,
            'slippage': slippage,
            'entry': entry_price,
            'exit': exit_price,
            'qty': qty
        }
        
        self.trades.append(trade_record)
        self.total_slippage += slippage
        
        if pnl > 0:
            self.wins += 1
            self.total_profit += pnl
        else:
            self.losses += 1
            self.total_loss += abs(pnl)
    
    def record_fill(self, success: bool):
        """Record order fill success/timeout"""
        if success:
            self.fill_successes += 1
        else:
            self.fill_timeouts += 1
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_slippage': 0.0,
                'fill_rate': 0.0
            }
        
        win_rate = self.wins / total_trades if total_trades > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        avg_win = self.total_profit / self.wins if self.wins > 0 else 0
        avg_loss = self.total_loss / self.losses if self.losses > 0 else 0
        avg_slippage = self.total_slippage / total_trades if total_trades > 0 else 0
        total_fill_attempts = self.fill_successes + self.fill_timeouts
        fill_rate = self.fill_successes / total_fill_attempts if total_fill_attempts > 0 else 0
        
        return {
            'total_trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_slippage_pct': avg_slippage * 100,
            'fill_rate': fill_rate,
            'total_pnl': self.total_profit - self.total_loss
        }
    
    def log_performance_stats(self):
        """Log current performance statistics"""
        stats = self.get_stats()
        if stats['total_trades'] > 0:
            logging.info(f"Performance Stats: Trades={stats['total_trades']}, Win Rate={stats['win_rate']:.1%}, "
                        f"Profit Factor={stats['profit_factor']:.2f}, Total P&L=${stats['total_pnl']:.2f}, "
                        f"Avg Slippage={stats['avg_slippage_pct']:.3f}%, Fill Rate={stats['fill_rate']:.1%}")

# -----------------------
# Alpaca API wrapper
# -----------------------
@dataclass
class AlpacaInterface:
    trading_client: TradingClient
    data_api: tradeapi.REST  # Use alpaca_trade_api for data fetching

    @classmethod
    def from_env(cls, api_key: str, api_secret: str, paper: bool = True):
        tc = TradingClient(api_key, api_secret, paper=paper)
        # Initialize alpaca_trade_api REST client for data fetching
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        data_api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        return cls(trading_client=tc, data_api=data_api)

    def get_account_equity(self) -> float:
        account = self.trading_client.get_account()
        # account.cash or account.equity depending on account type
        try:
            equity = float(account.equity)
        except Exception:
            equity = float(account.cash)
        return equity

    def fetch_recent_bars(self, symbol: str, limit: int = LOOKBACK_BARS, timeframe=BAR_TIMEFRAME) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: timestamp, open, high, low, close, volume
        Uses alpaca_trade_api.get_crypto_bars() which provides real-time data
        """
        try:
            # Calculate time range for fetching bars
            now = pd.Timestamp.now(tz='UTC')
            # Fetch last hour of data to ensure we get enough bars
            start = (now - pd.Timedelta('1 hour')).strftime('%Y-%m-%dT%H:%M:%SZ')
            end = now.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            logging.info(f"Requesting bars from {start} to {end} (now: {now})")
            
            # Use get_crypto_bars from alpaca_trade_api (works with paper trading)
            try:
                data = self.data_api.get_crypto_bars(symbol, TradeApiTimeFrame.Minute,
                                                     start=start, end=end)
                # Convert to DataFrame if needed
                if hasattr(data, 'df'):
                    df = data.df
                else:
                    df = data
            except AttributeError:
                # Fallback to get_bars if get_crypto_bars not available
                data = self.data_api.get_bars(symbol, TradeApiTimeFrame.Minute,
                                            start, end, adjustment='raw')
                df = data.df if hasattr(data, 'df') else data
            
            # Ensure we have the required columns and data types
            if len(df) > 0:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Limit to requested number of bars
                if len(df) > limit:
                    df = df.tail(limit)
                
                # Calculate how old the latest bar is
                latest_timestamp = df.index[-1]
                age_minutes = (now - latest_timestamp).total_seconds() / 60
                logging.info(f"Fetched {len(df)} bars. Latest timestamp: {latest_timestamp} (age: {age_minutes:.1f} minutes)")
            else:
                logging.warning(f"No bars returned for {symbol}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching bars: {e}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def place_limit_buy(self, symbol: str, qty: float, limit_price: float, tif: str = "gtc"):
        # OrderRequest wrappers (alpaca-py)
        # IMPORTANT: Crypto orders only support 'gtc' or 'ioc' time_in_force, NOT 'day'
        req = LimitOrderRequest(
            symbol=symbol,
            notional=None,
            qty=str(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,  # Use GTC for crypto (Good 'Til Canceled)
            limit_price=str(limit_price)
        )
        order = self.trading_client.submit_order(order_data=req)
        logging.info(f"Submitted limit buy: {order}")
        return order

    def place_market_buy(self, symbol: str, qty: float):
        # Market order for immediate fills (uses GTC for crypto)
        req = MarketOrderRequest(
            symbol=symbol,
            qty=str(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC  # Use GTC for crypto
        )
        order = self.trading_client.submit_order(order_data=req)
        logging.info(f"Submitted market buy: {order}")
        return order

    def place_market_sell(self, symbol: str, qty: float):
        # IMPORTANT: Crypto orders only support 'gtc' or 'ioc' time_in_force, NOT 'day'
        req = MarketOrderRequest(
            symbol=symbol,
            qty=str(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC  # Use GTC for crypto
        )
        order = self.trading_client.submit_order(order_data=req)
        logging.info(f"Submitted market sell: {order}")
        return order

    def cancel_order(self, order_id: str):
        return self.trading_client.cancel_order(order_id)

# -----------------------
# WebSocket Handler for Real-Time Updates
# -----------------------
class AlpacaWebSocketHandler:
    """Handles WebSocket connections for real-time trade updates and market data"""
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trade_updates_queue = Queue()
        self.crypto_data_stream = None
        
    # Note: TradingStream removed - we're using polling instead of WebSocket for trade updates

        
    def start_crypto_data_stream(self, symbol: str, on_bar_callback, on_quote_callback=None):
        """Start WebSocket stream for real-time crypto market data"""
        self.crypto_data_stream = CryptoDataStream(self.api_key, self.api_secret)
        
        async def bar_handler(bar):
            """Handle real-time bar updates"""
            logging.debug(f"Bar Update: {bar.symbol} - Close: {bar.close}, Volume: {bar.volume}")
            if on_bar_callback:
                on_bar_callback(bar)
        
        async def quote_handler(quote):
            """Handle real-time quote updates"""
            if on_quote_callback:
                on_quote_callback(quote)
        
        # Subscribe to bars and quotes for the symbol
        self.crypto_data_stream.subscribe_bars(bar_handler, symbol)
        if on_quote_callback:
            self.crypto_data_stream.subscribe_quotes(quote_handler, symbol)
        
        # Start stream in background thread
        import asyncio
        def run_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.crypto_data_stream._run_forever())
        
        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()
        logging.info(f"Crypto data WebSocket stream started for {symbol}")
    
    def stop(self):
        """Stop all WebSocket streams"""
        if self.trading_stream:
            self.trading_stream.stop()
        if self.crypto_data_stream:
            self.crypto_data_stream.stop()
        logging.info("WebSocket streams stopped")

# -----------------------
# Strategy manager
# -----------------------
class Mover:
    def __init__(self, alpaca: AlpacaInterface, symbol: str, profit_target: float = TARGET_USD_PER_TRADE, 
                 ema_fast: int = 9, ema_slow: int = 21, use_websocket: bool = True, buy_conditions: dict = None):
        self.alpaca = alpaca
        self.symbol = symbol
        self.fees_slippage = estimate_fees_and_slippage()
        self.target_per_trade = profit_target
        self.gross_pct = GROSS_PCT
        self.trades_per_10min = TRADES_PER_10MIN
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.use_websocket = use_websocket
        
        # Buy condition toggles (use provided or defaults from config)
        self.buy_conditions = buy_conditions if buy_conditions else BUY_CONDITIONS.copy()

        # runtime state
        self.entry_order_id: Optional[str] = None
        self.position_qty = 0.0
        self.position_entry_price = 0.0
        self.expected_entry_price = 0.0  # For slippage tracking
        self.session_start_equity = self.alpaca.get_account_equity()
        self.running_loss = 0.0
        
        # Trailing stop and partial exit state
        self.current_stop_price = 0.0
        self.partial_exit_taken = False
        self.breakeven_moved = False
        
        # Order timeout tracking
        self.order_placed_time = None
        self.order_timeout_seconds = 10
        
        # Performance tracking
        self.performance = PerformanceTracker()
        
        # WebSocket state (kept for trade updates only)
        self.ws_handler = None
        self.last_bar = None
        self.pending_fill = False
        
        if self.use_websocket:
            # Initialize WebSocket handler ONLY for trade updates (order fills)
            self.ws_handler = AlpacaWebSocketHandler(API_KEY, API_SECRET, paper=PAPER)
            # Start trade updates stream only
            self.ws_handler.start_trading_stream(self.on_trade_update)
            logging.info("WebSocket enabled for trade updates only (using REST API polling for price data)")
        else:
            logging.info("WebSocket disabled - using REST API polling for all data")

    def evaluate_signals_and_act(self):
        # Always fetch fresh data via REST API (polling method)
        # This ensures we get the latest data, especially important for paper trading
        df = self.alpaca.fetch_recent_bars(self.symbol, limit=LOOKBACK_BARS)
        
        # Get the timestamp of the most recent bar to verify fresh data
        latest_bar_time = df.index[-1]
        
        close = df['close']
        
        # Calculate all indicators
        ema_fast = ema(close, self.ema_fast)
        ema_slow = ema(close, self.ema_slow)
        current_vwap = float(vwap(df).iloc[-1])
        current_close = float(close.iloc[-1])
        current_ema_fast = float(ema_fast.iloc[-1])
        current_ema_slow = float(ema_slow.iloc[-1])
        current_rsi = float(rsi(close, 14).iloc[-1])
        current_atr = float(atr(df, 14).iloc[-1])
        
        # MACD for optional confirmation
        macd_line, signal_line, histogram = macd(close)
        current_macd_hist = float(histogram.iloc[-1])
        prev_macd_hist = float(histogram.iloc[-2])
        macd_turning_positive = current_macd_hist > 0 and prev_macd_hist <= 0

        # Display current market data prominently (always shows fresh polled data)
        logging.info(f"POLLING [{latest_bar_time}] | Price: ${current_close:.4f} | EMA{self.ema_fast}: ${current_ema_fast:.4f} | EMA{self.ema_slow}: ${current_ema_slow:.4f} | VWAP: ${current_vwap:.4f} | RSI: {current_rsi:.1f}")
        
        logging.debug(f"Price={current_close:.4f} VWAP={current_vwap:.4f} EMA{self.ema_fast}={current_ema_fast:.4f} "
                     f"EMA{self.ema_slow}={current_ema_slow:.4f} RSI={current_rsi:.2f} ATR={current_atr:.4f} "
                     f"MACD_Hist={current_macd_hist:.4f}")

        # Enhanced trend filter with VWAP
        long_trend = (current_ema_fast > current_ema_slow) and (current_close > current_vwap)
        short_trend = (current_ema_fast < current_ema_slow) and (current_close < current_vwap)

        # pullback definition: price is slightly below EMA_fast but above EMA_slow (configurable)
        pullback_pct = (current_ema_fast - current_close) / current_ema_fast

        # Build entry conditions based on toggles
        # Check each condition individually
        uptrend_ok = long_trend if self.buy_conditions['UPTREND'] else True
        good_entry_ok = (pullback_pct >= 0 and pullback_pct <= 0.005) if self.buy_conditions['GOOD_ENTRY'] else True
        rsi_ok = (current_rsi > 40 and current_rsi < 65) if self.buy_conditions['RSI_CHECK'] else True
        
        # Combine all enabled conditions
        can_long = uptrend_ok and good_entry_ok and rsi_ok
        
        # Display BUY/SELL conditions status
        if self.position_qty == 0:
            # Build status string showing which conditions are enabled/disabled
            conditions_status = []
            
            # UPTREND status
            if self.buy_conditions['UPTREND']:
                trend_status = "UPTREND [PASS]" if long_trend else "DOWNTREND [FAIL]"
                if not long_trend:
                    if current_ema_fast <= current_ema_slow:
                        trend_status += f" (EMA{self.ema_fast} must be > EMA{self.ema_slow})"
                    if current_close <= current_vwap:
                        trend_status += f" (Price must be > VWAP)"
            else:
                trend_status = "UPTREND [DISABLED]"
            conditions_status.append(trend_status)
            
            # GOOD_ENTRY status
            if self.buy_conditions['GOOD_ENTRY']:
                entry_status = "GOOD ENTRY [PASS]" if (pullback_pct >= 0 and pullback_pct <= 0.005) else f"WAITING [FAIL] (pullback={pullback_pct*100:.2f}%)"
            else:
                entry_status = "GOOD ENTRY [DISABLED]"
            conditions_status.append(entry_status)
            
            # RSI status
            if self.buy_conditions['RSI_CHECK']:
                rsi_status = "RSI [PASS]" if (current_rsi > 40 and current_rsi < 65) else f"RSI [FAIL] (need 40-65)"
            else:
                rsi_status = "RSI [DISABLED]"
            conditions_status.append(rsi_status)
            
            buy_signal = "BUY SIGNAL!" if can_long and not self.pending_fill else "No Buy Signal"
            
            logging.info(f"BUY CONDITIONS: {' | '.join(conditions_status)} → {buy_signal}")
        
        # Optional: add MACD confirmation (uncomment if desired)
        # can_long = can_long and (macd_turning_positive or current_macd_hist > 0)

        # Check for order timeout if we have a pending order
        if self.pending_fill and self.order_placed_time:
            elapsed = time.time() - self.order_placed_time
            if elapsed > self.order_timeout_seconds:
                logging.warning(f"Order timeout after {elapsed:.1f}s - canceling order {self.entry_order_id}")
                try:
                    self.alpaca.cancel_order(self.entry_order_id)
                    self.performance.record_fill(success=False)
                except Exception as e:
                    logging.error(f"Failed to cancel order: {e}")
                self.pending_fill = False
                self.entry_order_id = None
                self.order_placed_time = None

        # if we currently have no position and conditions met, try to enter
        if self.position_qty == 0 and can_long and not self.pending_fill:
            self.try_enter_long(current_close, current_atr)
        else:
            # check exit rules: trend flip, stops, trailing stops, and partial exits
            if self.position_qty > 0:
                current_profit_pct = (current_close - self.position_entry_price) / self.position_entry_price
                current_pnl = (current_close - self.position_entry_price) * self.position_qty
                
                # Display position status
                logging.info(f"POSITION: Entry=${self.position_entry_price:.4f} | Current=${current_close:.4f} | P&L=${current_pnl:.2f} ({current_profit_pct*100:+.2f}%) | Stop=${self.current_stop_price:.4f}")
                
                # PARTIAL EXIT: Take 50% profit at 0.6 × target
                partial_target_pct = self.gross_pct * 0.6
                if current_profit_pct >= partial_target_pct and not self.partial_exit_taken:
                    qty_to_exit = self.position_qty * 0.5
                    logging.info(f"Partial target reached ({partial_target_pct:.2%}): taking 50% profit")
                    try:
                        self.alpaca.place_market_sell(self.symbol, qty_to_exit)
                        self.partial_exit_taken = True
                        # Move stop to breakeven after partial exit
                        self.current_stop_price = self.position_entry_price
                        self.breakeven_moved = True
                        logging.info(f"Stop moved to breakeven: {self.current_stop_price:.4f}")
                    except Exception as e:
                        logging.error(f"Partial exit failed: {e}")
                
                # TRAILING STOP: Trail when profit > 0.5 × target
                trailing_threshold_pct = self.gross_pct * 0.5
                if current_profit_pct >= trailing_threshold_pct:
                    trailing_stop = current_close - (0.5 * current_atr)
                    if trailing_stop > self.current_stop_price:
                        self.current_stop_price = trailing_stop
                        logging.debug(f"Trailing stop updated to {trailing_stop:.4f}")
                
                # Check if trend flipped
                if not long_trend:
                    logging.info("Trend flipped against us: exiting immediately")
                    self.force_exit_market(current_close)
                # Check stop-loss (use trailing stop if set)
                elif current_close <= self.current_stop_price:
                    stop_type = "Trailing stop" if self.current_stop_price > self.position_entry_price else "Stop-loss"
                    logging.info(f"{stop_type} hit at {current_close:.4f} <= {self.current_stop_price:.4f}: exit")
                    self.force_exit_market(current_close)
                # Check full take-profit target
                elif current_profit_pct >= self.gross_pct:
                    logging.info(f"Full target reached ({current_profit_pct:.2%}): exit")
                    self.force_exit_market(current_close)

        # session loss check
        current_equity = self.alpaca.get_account_equity()
        loss_pct = (self.session_start_equity - current_equity) / self.session_start_equity
        if loss_pct >= SESSION_MAX_LOSS_PCT:
            logging.warning(f"Session loss {loss_pct:.2%} >= {SESSION_MAX_LOSS_PCT:.2%}. Halting trading for session.")
            raise SystemExit("Session stop-loss reached")

    def try_enter_long(self, current_price: float, current_atr: float):
        account_equity = self.alpaca.get_account_equity()
        try:
            usd_size = compute_position_size_usd(self.target_per_trade, self.gross_pct, self.fees_slippage, account_equity)
        except ValueError as e:
            logging.error(f"Position sizing error: {e}")
            return

        qty = usd_size / current_price
        # floor qty to acceptable precision (exchange dependent)
        qty = float(f"{qty:.6f}")

        # USE MARKET ORDER for immediate fills (better for testing/fast markets)
        # If you want limit orders, change use_market_order to False
        use_market_order = True
        
        if use_market_order:
            # Market order - fills immediately at current market price
            self.expected_entry_price = current_price  # Track for slippage calculation
            logging.info(f"Placing MARKET buy: qty={qty} USDsize={usd_size:.2f} at ~${current_price:.4f}")
            order = self.alpaca.place_market_buy(self.symbol, qty)
        else:
            # Limit order - waits for price to come to you (may not fill)
            limit_price = current_price * 0.9995  # 0.05% below market for better fill chance
            self.expected_entry_price = limit_price
            logging.info(f"Placing LIMIT buy: qty={qty} USDsize={usd_size:.2f} limit={limit_price:.4f}")
            order = self.alpaca.place_limit_buy(self.symbol, qty, limit_price)
        
        self.entry_order_id = order.id
        self.order_placed_time = time.time()  # Track order time for timeout
        
        if self.use_websocket:
            # With WebSocket, we'll get fill notification via trade_updates stream
            # Set pending flag and let the WebSocket handler process the fill
            self.pending_fill = True
            logging.info("Order submitted - waiting for WebSocket fill notification")
        else:
            # Fallback to polling method
            time_waited = 0
            while time_waited < 10:  # wait up to 10 seconds for a fill
                # get order status
                try:
                    ord_status = self.alpaca.trading_client.get_order_by_client_order_id(order.client_order_id)
                except Exception:
                    ord_status = None
                if ord_status and ord_status.filled_qty and float(ord_status.filled_qty) > 0:
                    filled_qty = float(ord_status.filled_qty)
                    avg_fill_price = float(ord_status.filled_avg_price) if ord_status.filled_avg_price else limit_price
                    self.on_enter(filled_qty, avg_fill_price)
                    return
                time.sleep(1.0)
                time_waited += 1
            # if not filled, cancel
            logging.info("Limit not filled in timeout, canceling order")
            try:
                self.alpaca.cancel_order(order.id)
            except Exception as exc:
                logging.debug("Cancel failed: %s", exc)
    
    def on_bar_update(self, bar):
        """Handle real-time bar updates from WebSocket"""
        # Log every bar received to verify WebSocket is working
        logging.info(f"WebSocket Bar Received: {bar.symbol} @ {bar.timestamp} - O:${bar.open:.4f} H:${bar.high:.4f} L:${bar.low:.4f} C:${bar.close:.4f} V:{bar.volume}")
        
        # Convert bar to dict format matching historical data
        bar_dict = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        # Add to buffer and maintain size
        self.bars_buffer.append(bar_dict)
        if len(self.bars_buffer) > LOOKBACK_BARS:
            self.bars_buffer.pop(0)  # Remove oldest bar
        
        # Switch to live data mode after first bar
        if not self.use_live_data and len(self.bars_buffer) >= LOOKBACK_BARS:
            self.use_live_data = True
            logging.info("Switched to live WebSocket data")
        
        # Store last bar for quick access
        self.last_bar = bar_dict
    
    def on_trade_update(self, trade_update):
        """Handle real-time trade updates from WebSocket"""
        event = trade_update.event
        order = trade_update.order
        
        logging.info(f"WebSocket Trade Update: {event} - Order: {order.id}, Status: {order.status}")
        
        # Handle fill events
        if event in ['fill', 'partial_fill']:
            if order.id == self.entry_order_id:
                filled_qty = float(order.filled_qty)
                avg_fill_price = float(order.filled_avg_price)
                self.on_enter(filled_qty, avg_fill_price)
                self.pending_fill = False
                self.entry_order_id = None
                self.order_placed_time = None
                self.performance.record_fill(success=True)
        
        # Handle order rejections or cancellations
        elif event in ['rejected', 'canceled', 'expired']:
            if order.id == self.entry_order_id:
                logging.warning(f"Entry order {event}: {order.id}")
                self.pending_fill = False
                self.entry_order_id = None
                self.order_placed_time = None
                self.performance.record_fill(success=False)

    def on_enter(self, qty: float, avg_price: float):
        """Called when position is entered"""
        self.position_qty = qty
        self.position_entry_price = avg_price
        
        # Initialize stop at entry - 1.5 * ATR (will be set properly in next evaluation)
        self.current_stop_price = avg_price * 0.985  # Temporary 1.5% stop
        self.partial_exit_taken = False
        self.breakeven_moved = False
        
        # Calculate slippage
        slippage = abs(avg_price - self.expected_entry_price) / self.expected_entry_price if self.expected_entry_price > 0 else 0
        slippage_bps = slippage * 10000  # basis points
        
        logging.info(f"Entered position qty={qty:.6f} entry={avg_price:.4f} (slippage={slippage_bps:.1f}bps)")

    def force_exit_market(self, current_price: float = None):
        """Exit position at market price"""
        if self.position_qty <= 0:
            logging.debug("No position to exit")
            return
        
        exit_qty = self.position_qty
        entry_price = self.position_entry_price
        
        try:
            self.alpaca.place_market_sell(self.symbol, exit_qty)
        except Exception as e:
            logging.error(f"Market exit failed: {e}")
            return
        
        # Get exit price (use provided current_price or fetch latest)
        if current_price is None:
            try:
                exit_price = self.alpaca.fetch_recent_bars(self.symbol, limit=1).iloc[-1]['close']
            except:
                exit_price = entry_price  # Fallback
        else:
            exit_price = current_price
        
        # Compute realized P&L
        pnl = (exit_price - entry_price) * exit_qty
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        self.running_loss += -pnl if pnl < 0 else 0.0
        
        # Record trade performance
        self.performance.record_trade(pnl, entry_price, exit_price, self.expected_entry_price, exit_qty)
        
        # Log exit with emoji based on profit/loss
        result_emoji = "profit" if pnl > 0 else "loss"
        logging.info(f"{result_emoji} Exited position at {exit_price:.4f}, P&L=${pnl:.2f} ({pnl_pct:+.2f}%)")
        
        # Log performance stats every 5 trades
        if len(self.performance.trades) % 5 == 0:
            self.performance.log_stats()
        
        # Reset position state
        self.position_qty = 0.0
        self.position_entry_price = 0.0
        self.expected_entry_price = 0.0
        self.current_stop_price = 0.0
        self.partial_exit_taken = False
        self.breakeven_moved = False

# -----------------------
# Main execution loop
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Alpaca Crypto Trading Bot (Mover Strategy)")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Trading symbol (e.g., XRP/USD, BTC/USD)")
    parser.add_argument("--profit_target", type=float, default=TARGET_USD_PER_TRADE, help="Target profit per trade in USD")
    parser.add_argument("--interval", type=str, default="1m", help="Bar interval (e.g., 1m, 5m, 1h)")
    parser.add_argument("--ema_fast", type=int, default=9, help="Fast EMA period")
    parser.add_argument("--ema_slow", type=int, default=21, help="Slow EMA period")
    parser.add_argument("--trade_qty", type=float, default=None, help="Fixed trade quantity (overrides profit target sizing)")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"], help="Trading mode: paper or live")
    parser.add_argument("--no-websocket", action="store_true", help="Disable WebSocket streaming (use REST API polling)")
    
    # Buy condition toggles
    parser.add_argument("--no-uptrend", action="store_true", help="Disable UPTREND check (buy without checking EMA/VWAP)")
    parser.add_argument("--no-good-entry", action="store_true", help="Disable GOOD_ENTRY check (buy without checking pullback)")
    parser.add_argument("--no-rsi", action="store_true", help="Disable RSI check (buy without checking RSI range)")
    
    args = parser.parse_args()
    
    # Parse interval to TimeFrame (using TradeApiTimeFrame from alpaca_trade_api)
    interval_map = {
        "1m": TradeApiTimeFrame.Minute,
        "5m": TradeApiTimeFrame(5, TradeApiTimeFrame.Minute),
        "15m": TradeApiTimeFrame(15, TradeApiTimeFrame.Minute),
        "1h": TradeApiTimeFrame.Hour,
        "1d": TradeApiTimeFrame.Day
    }
    bar_timeframe = interval_map.get(args.interval, TradeApiTimeFrame.Minute)
    
    # Build buy conditions dict from command-line arguments
    buy_conditions = {
        'UPTREND': not args.no_uptrend,      # Enabled by default, disabled with --no-uptrend
        'GOOD_ENTRY': not args.no_good_entry, # Enabled by default, disabled with --no-good-entry
        'RSI_CHECK': not args.no_rsi,        # Enabled by default, disabled with --no-rsi
    }
    
    # Determine paper vs live mode
    paper_mode = (args.mode == "paper")
    use_websocket = not args.no_websocket
    
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Set APCA_API_KEY and APCA_API_SECRET environment variables first")

    ws_status = "WebSocket (trade updates) + REST API (polling)" if use_websocket else "REST API (polling only)"
    
    # Log buy conditions status
    enabled_conditions = [k for k, v in buy_conditions.items() if v]
    disabled_conditions = [k for k, v in buy_conditions.items() if not v]
    conditions_status = f"Enabled: {', '.join(enabled_conditions) if enabled_conditions else 'None'}"
    if disabled_conditions:
        conditions_status += f" | Disabled: {', '.join(disabled_conditions)}"
    
    logging.info(f"Starting Mover strategy with symbol={args.symbol}, profit_target=${args.profit_target}, interval={args.interval}, mode={args.mode}, connection={ws_status}")
    logging.info(f"Buy Conditions: {conditions_status}")
    
    alpaca_interface = AlpacaInterface.from_env(API_KEY, API_SECRET, paper=paper_mode)
    mover = Mover(alpaca_interface, args.symbol, profit_target=args.profit_target, 
                  ema_fast=args.ema_fast, ema_slow=args.ema_slow, use_websocket=use_websocket,
                  buy_conditions=buy_conditions)

    try:
        while True:
            try:
                mover.evaluate_signals_and_act()
            except ValueError as v:
                logging.error("ValueError (likely sizing): %s", v)
            except Exception as e:
                logging.exception("Unhandled exception in evaluate loop: %s", e)
            
            # Adjust sleep time based on connection mode
            sleep_time = 10  # Poll every 10 seconds to avoid rate limits
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting.")
        if mover.ws_handler:
            mover.ws_handler.stop()


if __name__ == "__main__":
    main()
