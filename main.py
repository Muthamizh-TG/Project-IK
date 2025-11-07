# ============================================================================
# CRYPTO TRADING BOT - USAGE EXAMPLES
# ============================================================================
# Trade multiple cryptos with default $2000 lot size:
#   py main_polling.py BTCUSD ETHUSD XRPUSD --crypto 
#
# Trade single crypto with custom $1000 lot size:
#   py main_polling.py XRPUSD --crypto --lot 1000
# ===========================================================================
# this program logic get from shared code snippets @ 04/11/2025
# ============================================================================

import alpaca_trade_api as alpaca
import asyncio
import pandas as pd
import pytz
import sys
import logging
import time

from alpaca_trade_api.common import URL
from alpaca_trade_api.rest import TimeFrame

logger = logging.getLogger()

# ============================================================================
# ALPACA API CREDENTIALS (Paper Trading Account)
# ============================================================================
ALPACA_API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
ALPACA_SECRET_KEY = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"

# ============================================================================
# SCALP TRADING ALGORITHM CLASS
# ============================================================================
# This class implements a scalping trading strategy that:
# 1. Monitors price movements using 1-minute bars
# 2. Analyzes last 5 minutes to find optimal buy points (lowest price)
# 3. Places limit buy orders when conditions are met
# 4. Sells at profit when position is filled
# ============================================================================
class ScalpAlgo:
    def __init__(self, api, symbol, lot, crypto=False):
        """
        Initialize the trading algorithm for a specific symbol.
        
        Args:
            api: Alpaca REST API client
            symbol: Trading symbol (e.g., 'XRPUSD', 'BTCUSD')
            lot: Dollar amount to invest per trade (e.g., 2000 = $2000)
            crypto: True for crypto trading (24/7), False for stocks (market hours only)
        """
        self._api = api
        self._symbol = symbol  # Symbol for placing orders (e.g., 'XRPUSD')
        
        # ====================================================================
        # CRYPTO SYMBOL TRANSLATION
        # Different Alpaca API endpoints use different symbol formats:
        # - Data API needs: 'XRP/USD' (with slash)
        # - Trading API needs: 'XRPUSD' (no slash)
        # ====================================================================
        if crypto:
            s = symbol.upper()
            if '/' in s:
                self.api_symbol = s
            elif s.endswith('USD') and len(s) > 3:
                # Convert 'XRPUSD' -> 'XRP/USD' for data API
                self.api_symbol = s[:-3] + '/' + 'USD'
            else:
                # fallback: use the original uppercased symbol
                self.api_symbol = s
        else:
            self.api_symbol = symbol
        
        self._lot = lot  # Dollar amount per trade
        self._crypto = crypto  # Crypto vs stock flag
        self._bars = pd.DataFrame()  # Historical price bars storage
        self._l = logger.getChild(self._symbol)  # Logger for this symbol
        self._last_bar_timestamp = None  # Track last processed bar time
        
        # ====================================================================
        # TIMEZONE SETUP
        # Crypto: UTC (trades 24/7)
        # Stocks: America/New_York (NYSE market hours)
        # ====================================================================
        tz = 'UTC' if self._crypto else 'America/New_York'
        now_utc = pd.Timestamp.now(tz='UTC').floor('1min')
        if tz == 'UTC':
            now = now_utc
        else:
            now = now_utc.tz_convert(tz)
        
        # ====================================================================
        # FETCH INITIAL PRICE DATA
        # Get today's bars to start with historical context
        # ====================================================================
        today = now.strftime('%Y-%m-%d')
        tomorrow = (now + pd.Timedelta('1day')).strftime('%Y-%m-%d')
        
        max_attempts = 10
        attempt = 0
        while True:
            try:
                if self._crypto:
                    try:
                        data = api.get_crypto_bars(self.api_symbol, TimeFrame.Minute,
                                                  start=today, end=tomorrow)
                        data = getattr(data, 'df', data)
                    except AttributeError:
                        data = api.get_bars(self.api_symbol, TimeFrame.Minute,
                                            today, tomorrow,
                                            adjustment='raw').df
                else:
                    data = api.get_bars(self.api_symbol, TimeFrame.Minute, today,
                                        tomorrow, adjustment='raw').df
                break
            except Exception as e:
                attempt += 1
                self._l.warn(f'failed to fetch initial bars (attempt {attempt}/{max_attempts}): {e}')
                if attempt >= max_attempts:
                    print(f'Warning: could not fetch initial bars for {symbol} after {max_attempts} attempts; starting with empty bar history', flush=True)
                    data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                    break
                continue
        
        self._bars = data
        # ====================================================================
        # ENSURE NUMERIC DATA TYPES
        # Critical: All price columns must be numeric for calculations
        # ====================================================================
        if len(self._bars) > 0:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in self._bars.columns:
                    self._bars[col] = pd.to_numeric(self._bars[col], errors='coerce')
            self._last_bar_timestamp = self._bars.index[-1]
        
        # Initialize trading state machine
        self._init_state()
        
        # Log initial state for debugging
        self._l.info(f'Bot initialized in state: {self._state} (position: {self._position is not None}, order: {self._order is not None})')

    def _init_state(self):
        """
        Initialize the trading state machine by checking existing orders/positions.
        
        State Machine:
        - TO_BUY: Ready to place a buy order (no position, no pending orders)
        - BUY_SUBMITTED: Buy order placed, waiting for fill
        - TO_SELL: Position filled, ready to place sell order
        - SELL_SUBMITTED: Sell order placed, waiting for fill
        
        After sell fills, cycle back to TO_BUY
        """
        symbol = self._symbol
        # Check for any existing orders for this symbol
        order = [o for o in self._api.list_orders() if o.symbol == symbol]
        # Check for any existing positions for this symbol
        position = [p for p in self._api.list_positions()
                    if p.symbol == symbol]
        self._order = order[0] if len(order) > 0 else None
        self._position = position[0] if len(position) > 0 else None
        
        # Determine current state based on position and order status
        if self._position is not None:
            # We have a position (bought crypto/stock)
            if self._order is None:
                self._state = 'TO_SELL'  # No pending order, ready to sell
            else:
                self._state = 'SELL_SUBMITTED'  # Sell order pending
                if self._order.side != 'sell':
                    self._l.warn(
                        f'state {self._state} mismatch order {self._order}')
        else:
            # No position (haven't bought yet or sold already)
            if self._order is None:
                self._state = 'TO_BUY'  # No pending order, ready to buy
            else:
                self._state = 'BUY_SUBMITTED'  # Buy order pending
                if self._order.side != 'buy':
                    self._l.warn(
                        f'state {self._state} mismatch order {self._order}')

    def _now(self):
        """Get current time in appropriate timezone (UTC for crypto, NY for stocks)"""
        tz = 'UTC' if getattr(self, '_crypto', False) else 'America/New_York'
        now_utc = pd.Timestamp.now(tz='UTC')
        if tz == 'UTC':
            return now_utc
        return now_utc.tz_convert(tz)

    def _outofmarket(self):
        """
        Check if we're outside trading hours.
        Crypto: Always returns False (trades 24/7)
        Stocks: Returns True after 3:55 PM EST (near market close)
        """
        if getattr(self, '_crypto', False):
            return False
        return self._now().time() >= pd.Timestamp('15:55').time()

    def fetch_new_bars(self):
        """
        POLLING METHOD: Fetch new price bars since the last update.
        
        This is called every 5 seconds to get latest 1-minute price data.
        Replaces WebSocket streaming (which is deprecated for crypto).
        
        Process:
        1. Calculate time range: from last bar timestamp to now
        2. Fetch new bars from Alpaca API
        3. Process each new bar (check for duplicates)
        4. Update last_bar_timestamp
        """
        try:
            # Determine start time for fetching new bars
            if self._last_bar_timestamp is None:
                # First fetch: get last hour of data
                start = (self._now() - pd.Timedelta('1 hour')).strftime('%Y-%m-%dT%H:%M:%SZ')
            else:
                # Subsequent fetches: get bars since last known bar
                start = (self._last_bar_timestamp + pd.Timedelta('1 minute')).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            end = self._now().strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Fetch bars using appropriate API method
            if self._crypto:
                try:
                    data = self._api.get_crypto_bars(self.api_symbol, TimeFrame.Minute,
                                                     start=start, end=end)
                    data = getattr(data, 'df', data)
                except AttributeError:
                    data = self._api.get_bars(self.api_symbol, TimeFrame.Minute,
                                             start, end, adjustment='raw').df
            else:
                data = self._api.get_bars(self.api_symbol, TimeFrame.Minute,
                                         start, end, adjustment='raw').df
            
            if len(data) > 0:
                # Process new bars, checking for duplicates
                for idx, row in data.iterrows():
                    # Only process if:
                    # 1. Newer than last timestamp AND
                    # 2. Not already in our DataFrame (duplicate check)
                    if (self._last_bar_timestamp is None or idx > self._last_bar_timestamp) and \
                       (len(self._bars) == 0 or idx not in self._bars.index):
                        self.on_bar_data(idx, row)
                        self._last_bar_timestamp = idx
                        
        except Exception as e:
            self._l.error(f'Error fetching new bars: {e}')

    def on_bar_data(self, timestamp, bar_data):
        """
        Process a new price bar (1 minute of data).
        
        Bar contains: open, high, low, close, volume prices for 1 minute
        
        Process:
        1. Check for duplicates (skip if already processed)
        2. Create new DataFrame row with price data
        3. Append to historical bars (_bars)
        4. Log the new bar
        
        This builds up historical data needed for analysis (need 25 bars minimum)
        """
        try:
            # Convert timestamp to string for robust comparison
            ts_str = str(timestamp)
            
            # DUPLICATE CHECK: Skip if we already have this exact timestamp
            if len(self._bars) > 0:
                existing_timestamps = [str(ts) for ts in self._bars.index]
                if ts_str in existing_timestamps:
                    # Silently skip duplicate (don't log to avoid spam)
                    return
            
            # Create new bar as a DataFrame row
            new_bar = pd.DataFrame({
                'open': [float(bar_data['open'])],
                'high': [float(bar_data['high'])],
                'low': [float(bar_data['low'])],
                'close': [float(bar_data['close'])],
                'volume': [float(bar_data['volume'])],
            }, index=[timestamp])
            
            # Add to historical data
            if len(self._bars) == 0:
                self._bars = new_bar
            else:
                self._bars = pd.concat([self._bars, new_bar])
                # Ensure columns remain numeric after concat (critical for calculations)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    self._bars[col] = pd.to_numeric(self._bars[col], errors='coerce')
            
            # Log the new bar addition
            self._l.info(
                f'received bar start: {timestamp}, close: {bar_data["close"]}, len(bars): {len(self._bars)}')
            
        except Exception as e:
            self._l.error(f'Error in on_bar_data: {e}')
            import traceback
            traceback.print_exc()

    def checkup(self, position):
        """
        Periodic health check for orders and positions.
        
        Checks:
        1. Cancel buy orders that haven't filled after 2 minutes
        2. Force sell positions before market close (stocks only)
        """
        now = self._now()
        order = self._order
        
        # Cancel stale buy orders (older than 2 minutes)
        if (order is not None and
            order.side == 'buy' and now -
                order.submitted_at.tz_convert(tz='America/New_York') > pd.Timedelta('2 min')):
            # Get current price for logging
            if self._crypto and len(self._bars) > 0:
                last_price = float(self._bars['close'].iloc[-1])
            else:
                symbol_to_use = self.api_symbol if self._crypto else self._symbol
                last_price = self._api.get_latest_trade(symbol_to_use).price
            self._l.info(
                f'canceling missed buy order {order.id} at {order.limit_price} '
                f'(current price = {last_price})')
            self._cancel_order()

        # Force sell if we have a position and market is about to close (stocks only)
        if self._position is not None and self._outofmarket():
            self._submit_sell(bailout=True)

    def _cancel_order(self):
        """Cancel the current pending order"""
        if self._order is not None:
            self._api.cancel_order(self._order.id)

    def _calc_buy_signal(self):
        """
        ====================================================================
        CORE TRADING STRATEGY: 5-Minute Low Point Analysis
        ====================================================================
        
        Analyzes the last 5 minutes of price data to find optimal buy entry.
        
        Strategy Logic:
        1. Collect last 5 bars (5 minutes of price data)
        2. Calculate 20-period moving average (MA20) for trend context
        3. Find lowest and highest prices in the 5-minute window
        4. Calculate trend direction (UP or DOWN)
        5. Generate BUY signal when ALL conditions met:
           - Price is near MA20 (within 0.5% - means price is at average level)
           - Meaningful price movement (range > $0.001 - not flat/stagnant)
           - Current price near lowest point (within 0.2% - good entry)
        
        Returns:
            float: Target buy price (lowest price in window) if signal generated
            None: No buy signal
        
        Example:
            If last 5 minutes: [2.41, 2.40, 2.39, 2.38, 2.37]
            And MA20 = 2.39, current = 2.37
            Signal: BUY at 2.37 (lowest point, near MA, downward trend)
        ====================================================================
        """
        try:
            # Need at least 25 bars: 5 for analysis + 20 for moving average
            if len(self._bars) < 25:
                self._l.info(f'Not enough data for analysis: {len(self._bars)} bars (need 25+)')
                return None
            
            # Get last 5 minutes of closing prices
            last_5_bars = self._bars.tail(5)
            closes = pd.to_numeric(last_5_bars['close'], errors='coerce').values
            
            # Calculate 20-period moving average (trend baseline)
            mavg_20 = self._bars['close'].astype(float).rolling(20).mean().iloc[-1]
            current_price = float(self._bars['close'].iloc[-1])
            
            # Print MA20 at the top for monitoring
            print(f'[{self._symbol}] MA20: ${mavg_20:.6f} | Current Price: ${current_price:.6f}', flush=True)
            
            # Find price range in last 5 minutes
            lowest_price = float(closes.min())
            highest_price = float(closes.max())
            price_range = highest_price - lowest_price
            
            # Calculate trend: compare early prices vs recent prices
            early_avg = closes[:2].mean()  # First 2 minutes average
            recent_avg = closes[-2:].mean()  # Last 2 minutes average
            trend = "UP" if recent_avg > early_avg else "DOWN"
            
            # Log analysis for monitoring
            self._l.info(
                f'5-min analysis: lowest={lowest_price:.5f}, highest={highest_price:.5f}, '
                f'current={current_price:.5f}, MA20={mavg_20:.5f}, trend={trend}, range={price_range:.5f}'
            )
            
            # ================================================================
            # BUY SIGNAL CONDITIONS (all must be true)
            # ================================================================
            # 1. Price near moving average (within 0.5% tolerance)
            ma_threshold = mavg_20 * 0.005  # 0.5% of MA
            price_near_ma = abs(current_price - mavg_20) <= ma_threshold
            
            # 2. Meaningful price movement (not flat/stagnant)
            has_movement = price_range > 0.001
            
            # 3. Current price close to lowest (within 0.2% - good entry point)
            near_lowest = current_price <= (lowest_price * 1.002)
            
            if price_near_ma and has_movement and near_lowest:
                self._l.info(
                    f'BUY SIGNAL: Price {current_price:.5f} near MA {mavg_20:.5f}, '
                    f'at low point (lowest: {lowest_price:.5f}), trend: {trend}'
                )
                return lowest_price  # Return the target buy price (lowest in window)
            
            return None  # No signal - conditions not met
            
        except Exception as e:
            self._l.error(f'Error calculating buy signal: {e}')
            import traceback
            traceback.print_exc()
            return None

    def on_order_update(self, event, order):
        """
        Handle order status updates (fill, partial fill, cancel, reject).
        
        State transitions:
        - BUY fills -> transition to TO_SELL, submit sell order
        - SELL fills -> transition to TO_BUY (cycle repeats)
        - Cancel/Reject -> reset to appropriate state
        """
        self._l.info(f'order update: {event} = {order}')
        
        if event == 'fill':
            # Order completely filled
            self._order = None
            if self._state == 'BUY_SUBMITTED':
                # Buy filled - now we have a position
                self._position = self._api.get_position(self._symbol)
                self._transition('TO_SELL')
                self._submit_sell()  # Immediately place sell order for profit
                return
            elif self._state == 'SELL_SUBMITTED':
                # Sell filled - position closed
                self._position = None
                self._transition('TO_BUY')  # Ready to buy again
                return
                
        elif event == 'partial_fill':
            # Order partially filled (some quantity filled, rest pending)
            self._position = self._api.get_position(self._symbol)
            self._order = self._api.get_order(order['id'])
            return
            
        elif event in ('canceled', 'rejected'):
            # Order canceled or rejected
            if event == 'rejected':
                self._l.warn(f'order rejected: current order = {self._order}')
            self._order = None
            
            if self._state == 'BUY_SUBMITTED':
                # Buy order failed
                if self._position is not None:
                    # We have partial position - transition to sell
                    self._transition('TO_SELL')
                    self._submit_sell()
                else:
                    # No position - ready to try buying again
                    self._transition('TO_BUY')
                    
            elif self._state == 'SELL_SUBMITTED':
                # Sell order failed - try again
                self._transition('TO_SELL')
                self._submit_sell(bailout=True)  # Use market order to exit
            else:
                self._l.warn(f'unexpected state for {event}: {self._state}')

    def check_order_updates(self):
        """
        Poll for order status changes (replaces WebSocket order updates).
        
        Checks if pending order status changed:
        - filled -> call on_order_update('fill')
        - partially_filled -> call on_order_update('partial_fill')
        - canceled/expired/rejected -> call on_order_update with status
        """
        if self._order is not None:
            try:
                # Fetch latest order status from API
                updated_order = self._api.get_order(self._order.id)
                
                # Check if status changed
                if updated_order.status != self._order.status:
                    # Order status changed - handle the update
                    if updated_order.status == 'filled':
                        self.on_order_update('fill', {'id': updated_order.id, 'symbol': updated_order.symbol})
                    elif updated_order.status == 'partially_filled':
                        self.on_order_update('partial_fill', {'id': updated_order.id, 'symbol': updated_order.symbol})
                    elif updated_order.status in ('canceled', 'expired', 'rejected'):
                        self.on_order_update(updated_order.status, {'id': updated_order.id, 'symbol': updated_order.symbol})
                    
                    # Update or clear order reference
                    self._order = updated_order if updated_order.status not in ('filled', 'canceled', 'expired', 'rejected') else None
            except Exception as e:
                self._l.error(f'Error checking order updates: {e}')

    def _submit_buy(self, target_price=None):
        """
        ====================================================================
        SUBMIT BUY ORDER
        ====================================================================
        Place a limit buy order at the target price (from 5-min analysis).
        
        Args:
            target_price: Price to buy at (usually lowest price from analysis)
                         If None, uses current bar close price
        
        Process:
        1. Determine buy price (target or current)
        2. Calculate quantity:
           - CRYPTO: Fractional shares (e.g., 0.019 BTC)
             Formula: lot_size / price, rounded to 8 decimals
             Min: 0.0001 units
           - STOCKS: Whole shares only (e.g., 50 shares)
             Formula: int(lot_size / price)
             Min: 1 share
        3. Set time_in_force:
           - CRYPTO: 'gtc' (Good-Til-Canceled - stays open 24/7)
           - STOCKS: 'day' (expires at market close)
        4. Submit limit order to Alpaca
        5. Transition to BUY_SUBMITTED state
        
        Example:
            Lot: $1000, BTC price: $104,287
            Quantity: 1000 / 104287 = 0.00958769 BTC
            Order: Buy 0.00958769 BTC @ $104,287 (limit order)
        ====================================================================
        """
        # Determine the buy price
        if target_price is not None:
            buy_price = float(target_price)
            self._l.info(f'Using target price from 5-min analysis: {buy_price:.5f}')
        elif self._crypto and len(self._bars) > 0:
            # Use latest bar close price
            buy_price = float(self._bars['close'].iloc[-1])
        else:
            # Get latest trade price for stocks
            symbol_to_use = self.api_symbol if self._crypto else self._symbol
            trade = self._api.get_latest_trade(symbol_to_use)
            buy_price = float(trade.price)
        
        # ====================================================================
        # QUANTITY CALCULATION
        # Crypto: Supports fractional (0.00001 BTC)
        # Stocks: Whole shares only (1, 2, 3... shares)
        # ====================================================================
        if self._crypto:
            # Crypto supports fractional quantities
            amount = round(self._lot / buy_price, 8)  # Up to 8 decimal places
            # Ensure minimum quantity of 0.0001
            if amount < 0.0001:
                self._l.warn(f'Calculated quantity {amount} too small for ${self._lot} lot at ${buy_price:.2f}. Need larger lot size.')
                self._transition('TO_BUY')
                return
        else:
            # Stocks require whole shares
            amount = int(self._lot / buy_price)
            if amount < 1:
                self._l.warn(f'Calculated quantity {amount} < 1 share for ${self._lot} lot at ${buy_price:.2f}. Need larger lot size.')
                self._transition('TO_BUY')
                return
        
        # ====================================================================
        # TIME IN FORCE
        # Crypto: 'gtc' - Good-Til-Canceled (24/7 trading)
        # Stocks: 'day' - Expires at market close
        # ====================================================================
        time_in_force = 'gtc' if self._crypto else 'day'
        
        # Submit the order
        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side='buy',
                type='limit',  # Limit order (not market)
                qty=amount,
                time_in_force=time_in_force,
                limit_price=buy_price,
            )
        except Exception as e:
            self._l.error(f'Order submission failed: {e}')
            self._transition('TO_BUY')
            return

        self._order = order
        self._l.info(f'BUY ORDER PLACED: {amount} units @ ${buy_price:.5f} (total: ${amount * buy_price:.2f})')
        self._transition('BUY_SUBMITTED')

    def _submit_sell(self, bailout=False):
        """
        ====================================================================
        SUBMIT SELL ORDER
        ====================================================================
        Place a sell order to close the position at profit.
        
        Args:
            bailout: If True, use market order (immediate exit)
                    If False, use limit order (sell at profit)
        
        Process:
        1. Set time_in_force (gtc for crypto, day for stocks)
        2. If bailout: Use market order (sells immediately at current price)
        3. If normal: Use limit order at profit:
           - Get current price
           - Get cost basis (entry price)
           - Set limit: max(cost_basis + $0.01, current_price)
           - This ensures we sell at profit or break-even
        4. Submit sell order
        5. Transition to SELL_SUBMITTED state
        
        Example:
            Bought: 0.00958 BTC @ $104,287
            Current: $104,350
            Limit: max($104,287.01, $104,350) = $104,350
            Order: Sell 0.00958 BTC @ $104,350 (profit: $0.60)
        ====================================================================
        """
        # Crypto requires 'gtc' (Good-Til-Canceled), stocks use 'day'
        time_in_force = 'gtc' if self._crypto else 'day'
        
        # Base order parameters
        params = dict(
            symbol=self._symbol,
            side='sell',
            qty=self._position.qty,  # Sell entire position
            time_in_force=time_in_force,
        )
        
        if bailout:
            # BAILOUT MODE: Market order (immediate exit, any price)
            params['type'] = 'market'
        else:
            # NORMAL MODE: Limit order at profit
            # Get current price
            if self._crypto and len(self._bars) > 0:
                current_price = float(self._bars['close'].iloc[-1])
            else:
                symbol_to_use = self.api_symbol if self._crypto else self._symbol
                current_price = float(self._api.get_latest_trade(symbol_to_use).price)
            
            # Get cost basis (our entry price)
            cost_basis = float(self._position.avg_entry_price)
            
            # Set limit price: at least $0.01 profit, or current price if higher
            limit_price = max(cost_basis + 0.01, current_price)
            
            params.update(dict(
                type='limit',
                limit_price=limit_price,
            ))
            
        # Submit the sell order
        try:
            order = self._api.submit_order(**params)
        except Exception as e:
            self._l.error(e)
            self._transition('TO_SELL')
            return

        self._order = order
        self._l.info(f'submitted sell {order}')
        self._transition('SELL_SUBMITTED')

    def _transition(self, new_state):
        """Log and perform state transition"""
        self._l.info(f'transition from {self._state} to {new_state}')
        self._state = new_state

# ============================================================================
# MAIN FUNCTION: Setup and Run Trading Bot
# ============================================================================
async def main(args):
    """
    Main trading bot loop.
    
    Architecture:
    1. Initialize Alpaca API clients
    2. Create ScalpAlgo instance for each symbol
    3. Try WebSocket connection (for stocks)
    4. Fall back to polling mode (for crypto or if WebSocket fails)
    5. Run periodic loop every 5 seconds:
       - Fetch new bars
       - Check for buy signals
       - Place orders
       - Monitor order status
    """
    print('Initializing Alpaca clients (hybrid mode - WebSocket with polling fallback)...', flush=True)
    
    # ========================================================================
    # INITIALIZE ALPACA REST API CLIENT
    # Used for: placing orders, fetching data, checking account status
    # ========================================================================
    api = alpaca.REST(key_id=ALPACA_API_KEY,
                    secret_key=ALPACA_SECRET_KEY,
                    base_url="https://paper-api.alpaca.markets")
    print('REST API client created, checking connectivity...', flush=True)
    
    # Test connection
    try:
        acct = api.get_account()
        print(f"Connected to Alpaca account: {getattr(acct, 'id', 'unknown')} (status={getattr(acct, 'status', 'unknown')})", flush=True)
    except Exception as e:
        print('Warning: failed connectivity/account check:', e, flush=True)

    # ========================================================================
    # CREATE TRADING ALGORITHMS FOR EACH SYMBOL
    # Fleet: Dictionary of {symbol: ScalpAlgo instance}
    # Each symbol gets its own independent algorithm and state machine
    # ========================================================================
    fleet = {}
    symbols = args.symbols
    for symbol in symbols:
        print(f'Initializing algorithm for symbol: {symbol}', flush=True)
        try:
            algo = ScalpAlgo(api, symbol, lot=args.lot, crypto=getattr(args, 'crypto', False))
            key = algo.api_symbol if getattr(args, 'crypto', False) else symbol
            fleet[key] = algo
            print(f'Algorithm initialized for {symbol} (api symbol: {algo.api_symbol})', flush=True)
        except Exception as e:
            print(f'Error initializing algorithm for {symbol}: {e}', flush=True)
            import traceback
            traceback.print_exc()

    # ========================================================================
    # TRY WEBSOCKET CONNECTION (for stocks only)
    # Crypto WebSocket is deprecated, so we skip it for crypto trading
    # ========================================================================
    use_websocket = False
    conn = None
    
    # Skip WebSocket for crypto (known to be deprecated in this library version)
    if getattr(args, 'crypto', False):
        print('\nSkipping WebSocket - crypto uses polling mode', flush=True)
        print('(Alpaca crypto WebSocket streaming is deprecated in this library version)', flush=True)
    else:
        print('\nAttempting WebSocket connection for stock trading...', flush=True)
        try:
            conn = alpaca.Stream(key_id=ALPACA_API_KEY,
                               secret_key=ALPACA_SECRET_KEY,
                               base_url=URL("https://paper-api.alpaca.markets"),
                               data_feed='iex')
            
            # Set up bar handlers for WebSocket
            @conn.on_bar(*symbols)
            async def on_bars(bar):
                """Handle incoming bar data from WebSocket"""
                symbol = bar.symbol
                key = symbol if symbol in fleet else symbols[0]
                if key in fleet:
                    algo = fleet[key]
                    timestamp = bar.timestamp
                    bar_data = {
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    algo.on_bar_data(timestamp, bar_data)
                    
                    # Check for buy signals
                    if algo._state == 'TO_BUY' and len(algo._bars) >= 21:
                        signal = algo._calc_buy_signal()
                        if signal:
                            algo._submit_buy()
            
            use_websocket = True
            print('✓ WebSocket initialized successfully', flush=True)
                
        except Exception as e:
            print(f'✗ Failed to initialize WebSocket: {e}', flush=True)
            use_websocket = False
            conn = None

    # Display mode
    if use_websocket:
        print('\n' + '='*60)
        print('WEBSOCKET MODE: Real-time streaming data')
        print('Fastest reaction time with live bar updates')
        print('='*60 + '\n', flush=True)
    else:
        print('\n' + '='*60)
        print('POLLING MODE: Fetching data every 5 seconds')
        print('WebSocket unavailable - using reliable polling fallback')
        print('='*60 + '\n', flush=True)

    # ========================================================================
    # PERIODIC POLLING LOOP (runs every 5 seconds)
    # This is the main trading loop that:
    # 1. Fetches new price bars (if not using WebSocket)
    # 2. Analyzes data for buy signals
    # 3. Places orders when signals detected
    # 4. Monitors order status
    # 5. Performs health checks
    # ========================================================================
    async def periodic():
        """Main polling loop - runs every 5 seconds"""
        poll_count = 0
        while True:
            poll_count += 1
            
            # Check if market is open (stocks only, crypto runs 24/7)
            if not getattr(args, 'crypto', False):
                if not api.get_clock().is_open:
                    logger.info('exit as market is not open')
                    sys.exit(0)
            
            # Get current positions for all symbols
            positions = api.list_positions()
            
            # Process each symbol independently
            for symbol, algo in fleet.items():
                # ============================================================
                # FETCH NEW BARS (if not using WebSocket)
                # WebSocket sends bars automatically, polling fetches manually
                # ============================================================
                if not use_websocket:
                    algo.fetch_new_bars()
                    
                    # ========================================================
                    # CHECK FOR BUY SIGNALS
                    # Only when: state is TO_BUY and we have enough data
                    # ========================================================
                    if algo._state == 'TO_BUY' and len(algo._bars) >= 25:
                        # Get target buy price from 5-minute analysis
                        target_price = algo._calc_buy_signal()
                        if target_price is not None:
                            # Signal detected - place buy order
                            algo._submit_buy(target_price)
                
                # ============================================================
                # ALWAYS CHECK ORDER STATUS AND DO HEALTH CHECK
                # (needed even in WebSocket mode)
                # ============================================================
                algo.check_order_updates()
                pos = [p for p in positions if p.symbol == symbol]
                algo.checkup(pos[0] if len(pos) > 0 else None)
            
            # Wait 5 seconds before next poll
            await asyncio.sleep(5)

    async def websocket_runner():
        """
        Run WebSocket connection (for stocks only).
        If connection fails, falls back to polling mode.
        """
        try:
            await conn._run()
        except Exception as e:
            logger.error(f'WebSocket error: {e}')
            print(f'\nWebSocket connection lost: {e}', flush=True)
            print('Falling back to polling mode...', flush=True)
            nonlocal use_websocket
            use_websocket = False

    # ========================================================================
    # RUN THE BOT
    # Two modes:
    # 1. WebSocket + Periodic: For stocks (real-time data + polling checks)
    # 2. Periodic only: For crypto (polling only, no WebSocket)
    # ========================================================================
    try:
        if use_websocket and conn:
            # Run both WebSocket and periodic tasks in parallel
            await asyncio.gather(
                websocket_runner(),
                periodic()
            )
        else:
            # Run only periodic polling (crypto mode)
            await periodic()
    except KeyboardInterrupt:
        print('\nShutting down...', flush=True)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    import argparse

    # Setup logging (console and file)
    fmt = '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    
    # Log to file: console.log
    fh = logging.FileHandler('console.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    
    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    # ========================================================================
    # COMMAND LINE ARGUMENTS
    # ========================================================================
    # Usage examples:
    #   py main_polling.py XRPUSD --crypto --lot 1000
    #   py main_polling.py BTCUSD ETHUSD XRPUSD --crypto
    #   py main_polling.py AAPL TSLA --lot 5000  (stocks)
    # ========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('symbols', nargs='*',
                        help='One or more symbols to trade. If omitted defaults to XRP.')
    parser.add_argument('--lot', type=float, default=2000,
                        help='Dollar amount per trade (default: $2000)')
    parser.add_argument('--crypto', action='store_true',
                        help='Use crypto market (24/7 trading)')

    args = parser.parse_args()
    
    # Default to XRPUSD if no symbols provided
    if not getattr(args, 'symbols', None):
        args.symbols = ['XRPUSD']
        print('No symbols provided, defaulting to: XRPUSD')
        args.crypto = True

    print(f"Starting with symbols={args.symbols}, crypto={getattr(args, 'crypto', False)}, lot={args.lot}")

    # Run the bot
    try:
        asyncio.run(main(args))
    except Exception:
        import traceback
        print('Unhandled exception while running main():', flush=True)
        traceback.print_exc()
        sys.exit(1)
