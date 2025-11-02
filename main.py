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

ALPACA_API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
ALPACA_SECRET_KEY = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"


class ScalpAlgo:
    def __init__(self, api, symbol, lot, crypto=False):
        self._api = api
        self._symbol = symbol
        # Symbol used for API/data endpoints (may differ for crypto: e.g. 'XRP/USD')
        if crypto:
            s = symbol.upper()
            if '/' in s:
                self.api_symbol = s
            elif s.endswith('USD') and len(s) > 3:
                self.api_symbol = s[:-3] + '/' + 'USD'
            else:
                # fallback: use the original uppercased symbol
                self.api_symbol = s
        else:
            self.api_symbol = symbol
        self._lot = lot
        self._crypto = crypto
        self._bars = pd.DataFrame()
        self._l = logger.getChild(self._symbol)
        self._last_bar_timestamp = None
        
        # For crypto we use UTC and no market-open slicing (crypto trades 24/7)
        tz = 'UTC' if self._crypto else 'America/New_York'
        now_utc = pd.Timestamp.now(tz='UTC').floor('1min')
        if tz == 'UTC':
            now = now_utc
        else:
            now = now_utc.tz_convert(tz)
        
        # Request bars for initial window
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
        # Ensure all columns are numeric
        if len(self._bars) > 0:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in self._bars.columns:
                    self._bars[col] = pd.to_numeric(self._bars[col], errors='coerce')
            self._last_bar_timestamp = self._bars.index[-1]
        
        self._init_state()

    def _init_state(self):
        symbol = self._symbol
        order = [o for o in self._api.list_orders() if o.symbol == symbol]
        position = [p for p in self._api.list_positions()
                    if p.symbol == symbol]
        self._order = order[0] if len(order) > 0 else None
        self._position = position[0] if len(position) > 0 else None
        if self._position is not None:
            if self._order is None:
                self._state = 'TO_SELL'
            else:
                self._state = 'SELL_SUBMITTED'
                if self._order.side != 'sell':
                    self._l.warn(
                        f'state {self._state} mismatch order {self._order}')
        else:
            if self._order is None:
                self._state = 'TO_BUY'
            else:
                self._state = 'BUY_SUBMITTED'
                if self._order.side != 'buy':
                    self._l.warn(
                        f'state {self._state} mismatch order {self._order}')

    def _now(self):
        tz = 'UTC' if getattr(self, '_crypto', False) else 'America/New_York'
        now_utc = pd.Timestamp.now(tz='UTC')
        if tz == 'UTC':
            return now_utc
        return now_utc.tz_convert(tz)

    def _outofmarket(self):
        if getattr(self, '_crypto', False):
            return False
        return self._now().time() >= pd.Timestamp('15:55').time()

    def fetch_new_bars(self):
        """Fetch new bars since the last update (polling-based approach)"""
        try:
            # Fetch bars from the last known timestamp
            if self._last_bar_timestamp is None:
                start = (self._now() - pd.Timedelta('1 hour')).strftime('%Y-%m-%dT%H:%M:%SZ')
            else:
                start = (self._last_bar_timestamp + pd.Timedelta('1 minute')).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            end = self._now().strftime('%Y-%m-%dT%H:%M:%SZ')
            
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
                # Append new bars
                for idx, row in data.iterrows():
                    if self._last_bar_timestamp is None or idx > self._last_bar_timestamp:
                        self.on_bar_data(idx, row)
                        self._last_bar_timestamp = idx
                        
        except Exception as e:
            self._l.error(f'Error fetching new bars: {e}')

    def on_bar_data(self, timestamp, bar_data):
        """Process a new bar of data"""
        try:
            new_bar = pd.DataFrame({
                'open': [float(bar_data['open'])],
                'high': [float(bar_data['high'])],
                'low': [float(bar_data['low'])],
                'close': [float(bar_data['close'])],
                'volume': [float(bar_data['volume'])],
            }, index=[timestamp])
            
            # Ensure numeric dtypes when concatenating
            if len(self._bars) == 0:
                self._bars = new_bar
            else:
                self._bars = pd.concat([self._bars, new_bar])
                # Ensure columns remain numeric after concat
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    self._bars[col] = pd.to_numeric(self._bars[col], errors='coerce')
            
            self._l.info(
                f'received bar start: {timestamp}, close: {bar_data["close"]}, len(bars): {len(self._bars)}')
            
            if len(self._bars) < 21:
                return
            if self._outofmarket():
                return
            if self._state == 'TO_BUY':
                signal = self._calc_buy_signal()
                if signal:
                    self._submit_buy()
        except Exception as e:
            self._l.error(f'Error in on_bar_data: {e}')
            import traceback
            traceback.print_exc()

    def checkup(self, position):
        now = self._now()
        order = self._order
        if (order is not None and
            order.side == 'buy' and now -
                order.submitted_at.tz_convert(tz='America/New_York') > pd.Timedelta('2 min')):
            # For crypto, use the last bar's close price
            if self._crypto and len(self._bars) > 0:
                last_price = float(self._bars['close'].iloc[-1])
            else:
                symbol_to_use = self.api_symbol if self._crypto else self._symbol
                last_price = self._api.get_latest_trade(symbol_to_use).price
            self._l.info(
                f'canceling missed buy order {order.id} at {order.limit_price} '
                f'(current price = {last_price})')
            self._cancel_order()

        if self._position is not None and self._outofmarket():
            self._submit_sell(bailout=True)

    def _cancel_order(self):
        if self._order is not None:
            self._api.cancel_order(self._order.id)

    def _calc_buy_signal(self):
        try:
            # Ensure close column is numeric before calculation
            closes = pd.to_numeric(self._bars['close'], errors='coerce').values
            mavg = self._bars['close'].astype(float).rolling(20).mean().values
            
            if len(closes) >= 2 and len(mavg) >= 2:
                if not (pd.isna(closes[-2]) or pd.isna(closes[-1]) or pd.isna(mavg[-2]) or pd.isna(mavg[-1])):
                    if closes[-2] < mavg[-2] and closes[-1] > mavg[-1]:
                        self._l.info(
                            f'buy signal: closes[-2] {closes[-2]} < mavg[-2] {mavg[-2]} '
                            f'closes[-1] {closes[-1]} > mavg[-1] {mavg[-1]}')
                        return True
            self._l.info(
                f'closes[-2:] = {closes[-2:] if len(closes) >= 2 else closes}, mavg[-2:] = {mavg[-2:] if len(mavg) >= 2 else mavg}')
        except Exception as e:
            self._l.error(f'Error calculating buy signal: {e}')
        return False

    def on_order_update(self, event, order):
        self._l.info(f'order update: {event} = {order}')
        if event == 'fill':
            self._order = None
            if self._state == 'BUY_SUBMITTED':
                self._position = self._api.get_position(self._symbol)
                self._transition('TO_SELL')
                self._submit_sell()
                return
            elif self._state == 'SELL_SUBMITTED':
                self._position = None
                self._transition('TO_BUY')
                return
        elif event == 'partial_fill':
            self._position = self._api.get_position(self._symbol)
            self._order = self._api.get_order(order['id'])
            return
        elif event in ('canceled', 'rejected'):
            if event == 'rejected':
                self._l.warn(f'order rejected: current order = {self._order}')
            self._order = None
            if self._state == 'BUY_SUBMITTED':
                if self._position is not None:
                    self._transition('TO_SELL')
                    self._submit_sell()
                else:
                    self._transition('TO_BUY')
            elif self._state == 'SELL_SUBMITTED':
                self._transition('TO_SELL')
                self._submit_sell(bailout=True)
            else:
                self._l.warn(f'unexpected state for {event}: {self._state}')

    def check_order_updates(self):
        """Poll for order updates instead of using WebSocket"""
        if self._order is not None:
            try:
                updated_order = self._api.get_order(self._order.id)
                if updated_order.status != self._order.status:
                    # Order status changed
                    if updated_order.status == 'filled':
                        self.on_order_update('fill', {'id': updated_order.id, 'symbol': updated_order.symbol})
                    elif updated_order.status == 'partially_filled':
                        self.on_order_update('partial_fill', {'id': updated_order.id, 'symbol': updated_order.symbol})
                    elif updated_order.status in ('canceled', 'expired', 'rejected'):
                        self.on_order_update(updated_order.status, {'id': updated_order.id, 'symbol': updated_order.symbol})
                    
                    self._order = updated_order if updated_order.status not in ('filled', 'canceled', 'expired', 'rejected') else None
            except Exception as e:
                self._l.error(f'Error checking order updates: {e}')

    def _submit_buy(self):
        # For crypto, use the last bar's close price instead of get_latest_trade
        # which may not be available in all API versions
        if self._crypto and len(self._bars) > 0:
            current_price = float(self._bars['close'].iloc[-1])
        else:
            symbol_to_use = self.api_symbol if self._crypto else self._symbol
            trade = self._api.get_latest_trade(symbol_to_use)
            current_price = float(trade.price)
        
        amount = int(self._lot / current_price)
        
        # Crypto requires 'gtc' (Good-Til-Canceled), stocks use 'day'
        time_in_force = 'gtc' if self._crypto else 'day'
        
        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side='buy',
                type='limit',
                qty=amount,
                time_in_force=time_in_force,
                limit_price=current_price,
            )
        except Exception as e:
            self._l.info(e)
            self._transition('TO_BUY')
            return

        self._order = order
        self._l.info(f'submitted buy {order}')
        self._transition('BUY_SUBMITTED')

    def _submit_sell(self, bailout=False):
        # Crypto requires 'gtc' (Good-Til-Canceled), stocks use 'day'
        time_in_force = 'gtc' if self._crypto else 'day'
        
        params = dict(
            symbol=self._symbol,
            side='sell',
            qty=self._position.qty,
            time_in_force=time_in_force,
        )
        if bailout:
            params['type'] = 'market'
        else:
            # For crypto, use the last bar's close price
            if self._crypto and len(self._bars) > 0:
                current_price = float(self._bars['close'].iloc[-1])
            else:
                symbol_to_use = self.api_symbol if self._crypto else self._symbol
                current_price = float(self._api.get_latest_trade(symbol_to_use).price)
            
            cost_basis = float(self._position.avg_entry_price)
            limit_price = max(cost_basis + 0.01, current_price)
            params.update(dict(
                type='limit',
                limit_price=limit_price,
            ))
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
        self._l.info(f'transition from {self._state} to {new_state}')
        self._state = new_state


async def main(args):
    print('Initializing Alpaca clients (polling mode - no WebSocket)...', flush=True)
    
    api = alpaca.REST(key_id=ALPACA_API_KEY,
                    secret_key=ALPACA_SECRET_KEY,
                    base_url="https://paper-api.alpaca.markets")
    print('Alpaca clients created (will attempt a light connectivity check)...', flush=True)
    
    try:
        acct = api.get_account()
        print(f"Connected to Alpaca account: {getattr(acct, 'id', 'unknown')} (status={getattr(acct, 'status', 'unknown')})", flush=True)
    except Exception as e:
        print('Warning: failed connectivity/account check:', e, flush=True)

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

    print('\n' + '='*60)
    print('POLLING MODE: Fetching data every 30 seconds')
    print('Note: This is less real-time than WebSocket streaming')
    print('='*60 + '\n', flush=True)

    async def periodic():
        while True:
            # For equities exit when the market is closed; for crypto run continuously
            if not getattr(args, 'crypto', False):
                if not api.get_clock().is_open:
                    logger.info('exit as market is not open')
                    sys.exit(0)
            
            # Poll for new data and order updates
            positions = api.list_positions()
            for symbol, algo in fleet.items():
                # Fetch new bars
                algo.fetch_new_bars()
                
                # Check order updates
                algo.check_order_updates()
                
                # Regular checkup
                pos = [p for p in positions if p.symbol == symbol]
                algo.checkup(pos[0] if len(pos) > 0 else None)
            
            await asyncio.sleep(30)  # Poll every 30 seconds

    try:
        await periodic()
    except KeyboardInterrupt:
        print('\nShutting down...', flush=True)


if __name__ == '__main__':
    import argparse

    fmt = '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler('console.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('symbols', nargs='*',
                        help='One or more symbols to trade. If omitted defaults to XRP.')
    parser.add_argument('--lot', type=float, default=2000)
    parser.add_argument('--crypto', action='store_true', help='Use crypto market (24/7)')

    args = parser.parse_args()
    if not getattr(args, 'symbols', None):
        args.symbols = ['XRPUSD']
        print('No symbols provided, defaulting to: XRPUSD')
        args.crypto = True

    print(f"Starting with symbols={args.symbols}, crypto={getattr(args, 'crypto', False)}, lot={args.lot}")

    try:
        asyncio.run(main(args))
    except Exception:
        import traceback
        print('Unhandled exception while running main():', flush=True)
        traceback.print_exc()
        sys.exit(1)
