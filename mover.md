# Project Cape - Complete Trading Bot Documentation

## Table of Contents
1. Overview
2. WebSocket Implementation
3. Implementation Checklist
4. New Features Summary
5. Performance Tracking
6. Configuration Guide
7. Troubleshooting

---

## 1. OVERVIEW

### High-Level Strategy
Use fast trend filters (short EMAs + VWAP), only scalp with the trend on small pullbacks (RSI / momentum confirmation), use market orders for immediate fills or limit orders when configured, size positions so each small-percent scalp multiplied by number of trades hits the required target, and protect with ATR/EMA-based stops + trailing stop when the market turns.

### Core Indicators
- Trend filter: EMA(9) and EMA(21) - only take long scalps when EMA9 > EMA21 and price > VWAP
- Entry signal: wait for 1-3 candle pullback toward EMA9/20 and then RSI confirmation (rising above 45 for longs)
- Stop method: ATR-based (entry - 1.5 * ATR for longs). Trailing stop used once trade is 0.5x target to lock profit
- VWAP: Volume Weighted Average Price for trend confirmation
- MACD: Optional momentum confirmation

### Position Sizing Formula
Let:
- T = 10 minutes (time window)
- N = number of trades expected in each 10-min window
- G = gross percentage move target per trade (e.g., 0.8% = 0.008)
- F = round-trip fees + expected slippage as fraction (0.4% = 0.004)
- P = required net profit per trade in dollars = 2.4 / N
- net_pct = G - F (must be > 0)
- S = P / net_pct (position size in USD)

Example:
- N = 6 trades per 10 minutes
- P = 2.4 / 6 = 0.4 per trade
- G = 0.008 (0.8% gross)
- F = 0.004 (0.4% fees + slippage)
- net_pct = 0.008 - 0.004 = 0.004 (0.4% net profit)
- S = 0.40 / 0.004 = 100 USD position size

Position size is capped at 2% of account equity for risk management.

---

## 2. DATA FETCHING & REAL-TIME UPDATES

### What Was Changed
The trading bot now uses a **hybrid approach** with the older `alpaca-trade-api` library for data fetching, which provides better real-time data for paper trading accounts. The newer `alpaca-py` library is still used for trading operations.

### Key Features

**Real-Time Data Fetching**
- Uses `alpaca-trade-api` library's `get_crypto_bars()` method
- Fetches data with 1-2 minute delay (much better than 17+ hour delay from alpaca-py)
- Polls every 10 seconds for fresh market data
- Displays data age in logs to verify freshness
- Works reliably with paper trading accounts

**Dual Library Strategy**
- **alpaca-trade-api**: For fetching historical and recent price data
- **alpaca-py**: For placing orders and managing positions
- Best of both worlds: reliable data + modern trading API

**Data Freshness Monitoring**
- Logs the age of each bar fetch: "Latest timestamp: 2025-11-05 13:48:00+00:00 (age: 1.5 minutes)"
- Time range logging: "Requesting bars from 2025-11-05T12:48:44Z to 2025-11-05T13:48:44Z"
- Helps verify data is recent enough for trading decisions

### How to Use

Run with REST API polling (recommended for paper trading):
```bash
python mover.py --symbol XRP/USD --profit_target 0.4 --interval 1m --ema_fast 9 --ema_slow 21 --mode paper --no-websocket
```

The `--no-websocket` flag is recommended because:
1. Paper trading WebSocket data can be unreliable
2. REST API polling with alpaca-trade-api provides better data freshness
3. Simpler debugging and more predictable behavior

### Output Examples

Startup:
```
INFO Starting Mover strategy with symbol=XRP/USD, profit_target=$0.4, interval=1m, mode=paper, connection=REST API (polling only)
INFO WebSocket disabled - using REST API polling for all data
```

Data Fetching:
```
INFO Requesting bars from 2025-11-05T12:48:44Z to 2025-11-05T13:48:44Z (now: 2025-11-05 13:48:44.962826+00:00)
INFO Fetched 53 bars. Latest timestamp: 2025-11-05 13:47:00+00:00 (age: 1.7 minutes)
INFO POLLING [2025-11-05 13:47:00+00:00] | Price: $2.2593 | EMA9: $2.2560 | EMA21: $2.2529 | VWAP: $2.2445 | RSI: 67.5
```

### Architecture

**Components**

1. AlpacaInterface Class
   - Uses `tradeapi.REST` from alpaca-trade-api for data
   - Uses `TradingClient` from alpaca-py for orders
   - Hybrid approach for best performance

2. fetch_recent_bars() Method
   - Calculates time range (last hour of data)
   - Calls `data_api.get_crypto_bars()`
   - Converts to pandas DataFrame
   - Ensures numeric data types
   - Returns fresh data with age verification

3. Main Loop
   - Polls every 10 seconds
   - Fetches fresh bars
   - Evaluates indicators
   - Places orders when conditions met

### Benefits

Real-Time Data Approach:
- **Fresh data**: 1-2 minute delay vs 17+ hour delay
- **Reliable**: Works consistently with paper trading
- **Verifiable**: Logs show exact data age
- **Simple**: No complex WebSocket management needed
- **Debuggable**: Easy to see what data you're getting

### Data Quality

The bot now displays comprehensive market data on each poll:
```
INFO POLLING [timestamp] | Price: $X.XXXX | EMA9: $X.XXXX | EMA21: $X.XXXX | VWAP: $X.XXXX | RSI: XX.X
```

This allows you to:
- Verify indicators are calculating correctly
- See market conditions in real-time
- Debug entry/exit decisions
- Monitor data freshness

---

## 3. IMPLEMENTATION CHECKLIST

### IMPLEMENTED CORRECTLY (100%)

**Core Indicators**
- EMA(9) and EMA(21) for trend - Configurable via args
- RSI(14) for momentum - Used in entry conditions
- ATR(14) for stops - Used for stop-loss calculation
- Trend filter (EMA9 > EMA21) - Implemented
- VWAP indicator - Added for enhanced trend confirmation
- MACD histogram - Optional momentum confirmation

**Math & Position Sizing**
- P = 2.4 / N_target = 0.40
- G = 0.004 (0.4% target)
- F = fees + slippage = 0.004
- net_pct = G - F
- S = P / net_pct
- Cap at 2% equity

**Risk Management**
- Account risk cap per trade: Max 2% of equity
- Session stop-loss: 2% session loss
- ATR-based stop-loss: entry - 1.5 * ATR

**Entry Rules**
- Trend filter (EMA9 > EMA21 and price > VWAP) - **[TOGGLEABLE]**
- Pullback detection (0-0.5% from EMA) - **[TOGGLEABLE]**
- RSI confirmation (40 < RSI < 65) - **[TOGGLEABLE]**
- Market orders for immediate fills (default) OR limit orders at price * 0.9995

**Buy Condition Toggles** (NEW)
- Each condition can be independently enabled/disabled
- Allows testing which conditions are most effective
- Command-line flags: --no-uptrend, --no-good-entry, --no-rsi
- Logs show [PASS], [FAIL], or [DISABLED] for each condition

**Exit Rules**
- Take-profit at G% (0.8%)
- Stop-loss at 1.5*ATR
- Trend flip exit (EMA9 < EMA21 or price < VWAP)
- Trailing stop when profit > 0.5*G
- Partial exit at 0.6*G (50% of position)

**WebSocket Streaming**
- Real-time data via REST API polling (alpaca-trade-api library)
- Polls every 10 seconds for fresh market data
- 1-2 minute data delay (acceptable for trading)
- Data age verification in logs
- WebSocket option available but REST polling recommended for paper trading

**Order Management**
- Market orders for immediate fills (default)
- Limit orders available (configurable in code)
- Order placement with GTC (Good 'Til Canceled) for crypto
- Auto-cancel unfilled orders after timeout
- Slippage measurement on fills

---

## 4. NEW FEATURES SUMMARY

**Real-Time Data with alpaca-trade-api**
- Switched to older alpaca-trade-api library for data fetching
- Provides 1-2 minute fresh data vs 17+ hour delay
- Works reliably with paper trading accounts
- Hybrid approach: alpaca-trade-api for data, alpaca-py for trading
- Data age monitoring and logging
- Impact: Tradeable real-time data, accurate indicators

**Buy Condition Toggles**
- Independent enable/disable for each entry condition
- Three toggleable conditions:
  - UPTREND: EMA9 > EMA21 and Price > VWAP
  - GOOD_ENTRY: Price pullback within 0.5% of EMA9
  - RSI_CHECK: RSI between 40-65
- Command-line flags: --no-uptrend, --no-good-entry, --no-rsi
- Visual feedback: [PASS], [FAIL], [DISABLED] in logs
- Allows A/B testing of strategy components
- Impact: Identify which conditions improve performance

**Market Order Execution**
- Default to market orders for immediate fills
- No more waiting for limit orders to fill
- Configurable in code: `use_market_order = True/False`
- GTC (Good 'Til Canceled) time-in-force for crypto
- Limit orders still available if configured
- Impact: Faster execution, no missed opportunities

**Timeout Buy Feature** (NEW)
- Safety mechanism to ensure trading activity
- Triggers buy after 5 minutes without normal buy signal
- Two-tier profit targets:
  - Normal buys: 0.8% gross (0.4% net after fees)
  - Timeout buys: 0.6% gross (0.2% net after fees)
- Countdown timer shown in logs: "timeout buy in X.X min"
- Configuration: ENABLE_TIMEOUT_BUY, TIMEOUT_BUY_MINUTES, TIMEOUT_BUY_TARGET_PCT
- Special handling:
  - Ignores all entry conditions (UPTREND, GOOD_ENTRY, RSI)
  - Skips trend flip exit rule (holds position regardless of trend)
  - Only exits at profit target or stop loss
- Impact: Prevents bot from sitting idle, generates activity even in downtrends

**Paper Trading Simulation** (NEW)
- Simulates instant order fills in paper trading mode
- Solves Alpaca's crypto paper trading fill issues
- Configuration: SIMULATE_FILLS = True (default)
- When enabled:
  - Market orders "fill" instantly at current price
  - No stuck orders accumulating
  - Clean position tracking
  - Works like live trading but with simulated execution
- When disabled: Uses actual Alpaca paper trading orders (may not fill)
- Impact: Reliable testing without order execution problems

**Enhanced Logging**
- Shows data timestamp and age on every poll
- Clear [PASS]/[FAIL]/[DISABLED] status for each condition
- Position status with entry, current price, P&L, and stop
- Request time ranges for debugging
- Impact: Better visibility into bot decisions

**VWAP Indicator**
- Volume Weighted Average Price - critical trend filter
- Integrated into trend filter: longs require price > VWAP
- Logs VWAP value in every poll
- Formula: Cumulative(Typical Price × Volume) / Cumulative(Volume)
- Impact: More accurate trend detection, reduces false signals

**Trailing Stops**
- Automatically move stop-loss to lock in profits
- Activates when profit reaches 0.5 × target (0.4%)
- Trails at 0.5 × ATR below current price
- Automatically updates as price moves up
- Logs trailing stop updates
- Impact: Locks in gains, lets winners run

**Layered Exits / Partial Profit Taking**
- Take 50% profit at 0.6 × target (0.48%)
- Rest exits at full target (0.8%)
- Monitors profit percentage continuously
- Moves stop to breakeven after partial exit
- Impact: Reduces risk, improves win rate

**Slippage Tracking**
- Measure actual fill price vs expected price
- Tracks expected_entry_price when placing order
- Calculates actual slippage on fill
- Records slippage for each trade
- Logs slippage in basis points (bps)
- Shows average slippage in performance stats
- Impact: Real-world cost tracking, fee optimization

**Performance Metrics**
- New PerformanceTracker class
- Tracks all trades with full details
- Calculates:
  - Win rate (wins / total trades)
  - Profit factor (total profit / total loss)
  - Average win/loss amounts
  - Average slippage percentage
  - Fill rate (successful fills / attempts)
  - Total P&L
- Logs stats every 5 trades
- Impact: Data-driven strategy optimization

**MACD Histogram**
- Momentum confirmation indicator
- Returns MACD line, signal line, histogram
- Detects MACD turning positive
- Can be enabled as entry filter (currently commented out)
- Optional usage for additional confirmation

**Improved Fee/Slippage Estimates**
- More realistic Alpaca crypto fees
- Maker fee: 0.15% per trade
- Slippage: 0.05% per trade
- Round-trip total: 0.4% (both sides)
- Properly documented in code
- Impact: Accurate profitability calculations

---

## 5. PERFORMANCE TRACKING

### Metrics Collected

**Per-Trade Metrics**
- Entry price and exit price
- Position quantity
- Profit/Loss in USD
- Profit/Loss percentage
- Slippage (actual vs expected entry)
- Trade duration (implicit)

**Aggregate Metrics**
- Total number of trades
- Number of wins and losses
- Win rate percentage
- Profit factor
- Average win amount
- Average loss amount
- Average slippage percentage
- Fill rate (successful fills / total attempts)
- Total P&L across all trades

### Performance Stats Output

Example output every 5 trades:
```
Performance Stats: Trades=10, Win Rate=70.0%, Profit Factor=2.45, 
Total P&L=$3.80, Avg Slippage=0.015%, Fill Rate=85.0%
```

### Using Performance Data

The PerformanceTracker class provides:
- Real-time performance monitoring
- Historical trade analysis
- Slippage cost analysis
- Order execution quality metrics

Access stats programmatically:
```python
stats = mover.performance.get_stats()
print(f"Win Rate: {stats['win_rate']:.1%}")
print(f"Profit Factor: {stats['profit_factor']:.2f}")
print(f"Total P&L: ${stats['total_pnl']:.2f}")
```

---

## 6. CONFIGURATION GUIDE

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --symbol | str | XRP/USD | Trading pair (e.g., XRP/USD, BTC/USD) |
| --profit_target | float | 0.40 | Target profit per trade in USD |
| --interval | str | 1m | Bar interval (1m, 5m, 15m, 1h, 1d) |
| --ema_fast | int | 9 | Fast EMA period |
| --ema_slow | int | 21 | Slow EMA period |
| --trade_qty | float | None | Fixed trade quantity (overrides profit target) |
| --mode | str | paper | Trading mode: paper or live |
| --no-websocket | flag | False | Disable WebSocket (use REST API - recommended) |
| --no-uptrend | flag | False | Disable UPTREND check (buy without checking EMA/VWAP) |
| --no-good-entry | flag | False | Disable GOOD_ENTRY check (buy without checking pullback) |
| --no-rsi | flag | False | Disable RSI check (buy without checking RSI range) |

### Configuration Constants

In mover.py:
```python
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
PAPER = True
SIMULATE_FILLS = True  # Simulate instant fills in paper mode (recommended)
SYMBOL = "XRP/USD"
BAR_TIMEFRAME = TradeApiTimeFrame.Minute
LOOKBACK_BARS = 100
TARGET_USD_PER_TRADE = 0.40
TRADES_PER_10MIN = 6
GROSS_PCT = 0.008  # 0.8% target (must be > fees+slippage)
MAX_ACCOUNT_RISK_PCT = 0.02
SESSION_MAX_LOSS_PCT = 0.02

# Buy condition toggles (can override with command-line)
BUY_CONDITIONS = {
    'UPTREND': True,      # EMA9 > EMA21 and Price > VWAP
    'GOOD_ENTRY': True,   # Price pullback within 0.5% of EMA9
    'RSI_CHECK': True,    # RSI between 40-65
}

# Timeout buy settings (NEW)
ENABLE_TIMEOUT_BUY = True           # Enable timeout buy feature
TIMEOUT_BUY_MINUTES = 5             # Minutes to wait before timeout buy
TIMEOUT_BUY_TARGET_PCT = 0.002      # 0.2% net profit target (excluding fees)
```

### Example Usage

**Standard trading (all conditions enabled):**
```bash
python mover.py --mode paper --no-websocket
```

**Test without RSI requirement:**
```bash
python mover.py --mode paper --no-websocket --no-rsi
```

**Test with only UPTREND check:**
```bash
python mover.py --mode paper --no-websocket --no-good-entry --no-rsi
```

**Test timeout buy feature (disable all conditions):**
```bash
python mover.py --mode paper --no-websocket --no-uptrend --no-good-entry --no-rsi
```
*This will trigger a timeout buy within 5 minutes and exit at 0.2% net profit*

### Adjusting Strategy Parameters

**Enable/Disable Simulated Fills:**
In mover.py around line 36:
```python
SIMULATE_FILLS = True   # Enable simulated fills (recommended for paper trading)
SIMULATE_FILLS = False  # Use actual Alpaca orders (may not fill in paper mode)
```

**Configure Timeout Buy:**
In mover.py around line 74-76:
```python
**Configure Timeout Buy:**
In mover.py around line 74-76:
```python
ENABLE_TIMEOUT_BUY = True    # Enable/disable timeout buy feature
TIMEOUT_BUY_MINUTES = 5      # How long to wait before timeout buy (default: 5 min)
TIMEOUT_BUY_TARGET_PCT = 0.002  # Net profit target for timeout buys (default: 0.2%)
```

**Switch to Limit Orders** (from Market Orders):
In mover.py around line 689, change:
```python
use_market_order = False  # Use limit orders instead
```

**Adjust Limit Order Aggressiveness**:
Line 625, change the multiplier:
```python
limit_price = current_price * 0.999  # 0.1% below market (more aggressive)
```

**Enable MACD Confirmation** (Optional):
In mover.py around line 520, uncomment:
```python
can_long = can_long and (macd_turning_positive or current_macd_hist > 0)
```

**Adjust Trailing Stop Distance**:
Line 570, change the multiplier:
```python
trailing_stop = current_close - (0.75 * current_atr)  # Tighter trailing
```

**Change Partial Exit Level**:
Line 560, change the percentage:
```python
partial_target_pct = self.gross_pct * 0.7  # Take 50% at 70% of target
```

**Modify Order Timeout**:
In __init__ method:
```python
self.order_timeout_seconds = 15  # Wait 15 seconds before canceling
```

**Adjust Profit Target**:
Change GROSS_PCT in configuration:
```python
GROSS_PCT = 0.010  # 1.0% target (more room above fees)
```

---

## 7. TROUBLESHOOTING

### Common Issues

**Position Sizing Error**
Error: "Position sizing error: net_pct <= 0 (gross=0.004, fees+slippage=0.004); strategy not viable"
Solution:
- Increase GROSS_PCT (e.g., 0.008 or 0.010)
- Reduce fees/slippage estimate in estimate_fees_and_slippage()
- Ensure: GROSS_PCT > fees + slippage

**Invalid Crypto Time-In-Force**
Error: "invalid crypto time_in_force"
Solution:
- Already fixed in current version
- Crypto orders use TimeInForce.GTC (not DAY)
- Applies to both market and limit orders

**Old/Delayed Data**
Issue: Data timestamp is hours old
Solution:
- Already fixed with alpaca-trade-api library
- Verify you see "age: 1-2 minutes" in logs
- If still delayed, check API credentials

**Symbol Format Error**
Error: "invalid symbol: XRPUSD does not match pattern"
Solution: Use format with slash: XRP/USD, BTC/USD

**No Bars Returned**
Error: "No bars returned for symbol"
Solution: 
- Check symbol is valid for crypto trading
- Verify market is open (crypto trades 24/7)
- Check API credentials are correct

**Order Not Filling (Market Orders)**
Issue: Market orders not executing
Solution:
- Check account has sufficient funds
- Verify symbol is tradeable in paper/live mode
- Check logs for API errors

**Order Not Filling (Limit Orders)**
Issue: Limit orders timing out
Solution:
- Switch to market orders (set use_market_order = True)
- Adjust limit price closer to market: current_price * 0.9995
- Increase timeout: self.order_timeout_seconds = 15
- Check order book depth

**Buy Signal Not Triggering**
Issue: Conditions never all met
Solution:
- Check which condition is failing in logs: [PASS] vs [FAIL]
- Disable specific conditions to test: --no-rsi, --no-good-entry, etc.
- Review market conditions (may not match entry criteria)
- Adjust RSI range or pullback percentage if too strict

**High Slippage**
Issue: Average slippage > 0.05%
Solution:
- Use smaller position sizes
- Trade during high liquidity periods
- Use limit orders instead of market orders

**Low Fill Rate**
Issue: Many orders timing out (limit orders)
Solution:
- Switch to market orders
- Increase order timeout
- Use more aggressive limit prices
- Check market volatility

**Session Stop-Loss Hit Quickly**
Issue: Trading halted due to 2% loss
Solution:
- Review entry conditions (may be too loose)
- Reduce position size
- Adjust stop-loss distance
- Check market conditions (high volatility?)
- Test with disabled conditions to identify issues

### Debug Mode

Enable detailed logging:
In mover.py:
```python
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
```

This shows:
- Current price and all indicators
- VWAP, EMA, RSI, ATR, MACD values
- Order book depth (if implemented)
- All API requests and responses

### Checking Connection Status

REST API polling shows:
```
INFO WebSocket disabled - using REST API polling for all data
INFO Requesting bars from 2025-11-05T12:48:44Z to 2025-11-05T13:48:44Z
INFO Fetched 53 bars. Latest timestamp: 2025-11-05 13:47:00+00:00 (age: 1.7 minutes)
```

Verify data is fresh:
1. Check "age: X minutes" is less than 5 minutes
2. Verify timestamp is recent (within last few minutes)
3. Ensure polling happens every 10 seconds
4. Look for "Requesting bars" messages

### Buy Condition Debugging

Use condition toggles to isolate issues:

**Test each condition individually:**
```bash
# Test only UPTREND
python mover.py --mode paper --no-websocket --no-good-entry --no-rsi

# Test only GOOD_ENTRY  
python mover.py --mode paper --no-websocket --no-uptrend --no-rsi

# Test only RSI
python mover.py --mode paper --no-websocket --no-uptrend --no-good-entry
```

**Logs show which condition is blocking:**
```
INFO BUY CONDITIONS: UPTREND [PASS] | GOOD ENTRY [FAIL] (pullback=-0.24%) | RSI [PASS] → No Buy Signal
```

In this example, GOOD_ENTRY is failing because pullback is -0.24% (price too far above EMA9).

### Performance Issues

**Bot Running Slowly**
- Reduce LOOKBACK_BARS (from 100 to 50)
- Increase sleep interval (from 5 to 10 seconds)
- Disable DEBUG logging

**High API Usage**
- Enable WebSocket mode (default)
- Increase check interval
- Reduce unnecessary bar fetches

---

## APPENDIX: EXPECTED OUTPUT EXAMPLES

### Startup Sequence
```
INFO Starting Mover strategy with symbol=XRP/USD, profit_target=$0.4, interval=1m, mode=paper, connection=REST API (polling only)
INFO Buy Conditions: Enabled: UPTREND, GOOD_ENTRY, RSI_CHECK | Disabled: None
INFO WebSocket disabled - using REST API polling for all data
```

### Data Fetching (Every 10 Seconds)
```
INFO Requesting bars from 2025-11-05T12:48:44Z to 2025-11-05T13:48:44Z (now: 2025-11-05 13:48:44.962826+00:00)
INFO Fetched 53 bars. Latest timestamp: 2025-11-05 13:47:00+00:00 (age: 1.7 minutes)
INFO POLLING [2025-11-05 13:47:00+00:00] | Price: $2.2593 | EMA9: $2.2560 | EMA21: $2.2529 | VWAP: $2.2445 | RSI: 67.5
```

### Condition Checking (No Signal)
```
INFO BUY CONDITIONS: UPTREND [PASS] | GOOD ENTRY [FAIL] (pullback=-0.24%) | RSI [FAIL] (need 40-65) → No Buy Signal
```

### Condition Checking (Signal Found)
```
INFO BUY CONDITIONS: UPTREND [PASS] | GOOD ENTRY [PASS] | RSI [PASS] → BUY SIGNAL!
```

### With Conditions Disabled
```
INFO Buy Conditions: Enabled: UPTREND | Disabled: GOOD_ENTRY, RSI_CHECK
INFO BUY CONDITIONS: UPTREND [PASS] | GOOD ENTRY [DISABLED] | RSI [DISABLED] → BUY SIGNAL!
```

### Entry Signal (Market Order)
```
INFO Placing MARKET buy: qty=44.173514 USDsize=100.00 at ~$2.2638
INFO Submitted market buy: Order(id='71672b92', symbol='XRP/USD', qty='44.173514', ...)
INFO Entered position qty=44.173514 entry=2.2640 (slippage=0.9bps)
```

### Entry Signal (Limit Order)
```
INFO Placing LIMIT buy: qty=44.220883 USDsize=100.00 limit=2.2591
INFO Submitted limit buy: Order(id='ea479c29', ...)
INFO Limit not filled in timeout, canceling order
```

### Position Monitoring
```
INFO POSITION: Entry=$2.2640 | Current=$2.2685 | P&L=$0.20 (+0.20%) | Stop=$2.2600
```

### Partial Exit
```
INFO Partial target reached (0.48%): taking 50% profit
INFO Stop moved to breakeven: 2.2640
INFO Exited position at 2.2749, P&L=$0.48 (+0.48%)
```

### Trailing Stop
```
DEBUG Trailing stop updated to 2.2700
DEBUG Trailing stop updated to 2.2715
INFO Trailing stop hit at 2.2713 <= 2.2715: exit
INFO Exited position at 2.2713, P&L=$0.64 (+0.32%)
```

### Full Target Exit
```
INFO Full target reached (0.80%): exit
INFO Exited position at 2.2821, P&L=$0.80 (+0.80%)
```

### Performance Stats (Every 5 Trades)
```
INFO Performance Stats: Trades=5, Win Rate=80.0%, Profit Factor=3.25, 
Total P&L=$1.85, Avg Slippage=0.018%, Fill Rate=100.0%
```

### Stop-Loss Hit
```
INFO Stop-loss hit at 2.2595 <= 2.2600: exit
INFO Exited position at 2.2595, P&L=$-0.20 (-0.20%)
```

### Trend Flip Exit
```
INFO Trend flipped against us: exiting immediately
INFO Exited position at 2.2650, P&L=$0.04 (+0.04%)
```

### Session Stop-Loss
```
WARNING Session loss 2.05% >= 2.00%. Halting trading for session.
```

### Simulated Fill (Paper Trading)
```
INFO Placing MARKET buy: qty=42.735043 USDsize=100.00 at ~$2.3400
INFO SIMULATED FILL: Paper trading mode - simulating instant fill at $2.3400
INFO Entered position qty=42.735043 entry=2.3400 (slippage=0.0bps)
```

### Timeout Buy
```
INFO BUY CONDITIONS: DOWNTREND [FAIL] | GOOD ENTRY [PASS] | RSI [FAIL] → TIMEOUT BUY! (waited 5.1 min)
INFO TIMEOUT BUY TRIGGERED: No buy signal for 5 minutes - buying at current price
INFO SIMULATED FILL: Paper trading mode - simulating instant fill at $2.3400
INFO POSITION: Entry=$2.3400 | Current=$2.3415 | P&L=$0.64 (+0.64%) | Stop=$2.3049 | Target=timeout target (0.2% net)
INFO Timeout buy target reached (0.64%): exit
```

---

## COMMON ISSUES & SOLUTIONS

### Problem: Orders Not Filling in Paper Trading

**Symptoms:**
```
INFO Placing MARKET buy: qty=42.363906 USDsize=100.00 at ~$2.3605
WARNING Market order not filled after 30s - checking final status
ERROR Market order failed to fill after 30s: status=OrderStatus.NEW
```

**Solution:**
Enable simulated fills (recommended for paper trading):
```python
SIMULATE_FILLS = True  # In mover.py line 36
```

Alpaca's crypto paper trading has unreliable order execution. Simulated fills provide instant execution for testing.

### Problem: Timeout Buys Exit Immediately

**Symptoms:**
```
INFO TIMEOUT BUY TRIGGERED: No buy signal for 5 minutes - buying at current price
INFO Entered position qty=42.735043 entry=2.3400
INFO Trend flipped against us: exiting immediately
INFO Exited position at 2.3400, P&L=$0.00 (+0.00%)
```

**Explanation:**
This was fixed in latest version. Timeout buys now ignore trend flip exits since they're designed to trade in any condition.

**If still occurring:**
Check line 643 in mover.py:
```python
if not long_trend and not self.timeout_buy_active:  # Should include "and not self.timeout_buy_active"
```

### Problem: Too Many Stuck Orders

**Symptoms:**
```
INFO Skipping buy - already have 43 open order(s)
```

**Solution:**
Run the cleanup script:
```bash
python cancel_all_orders.py
```

Or enable simulated fills to prevent stuck orders:
```python
SIMULATE_FILLS = True  # In mover.py
```

### Problem: Data Age Too Old

**Symptoms:**
```
INFO Fetched 50 bars. Latest timestamp: 2025-11-05 08:30:00+00:00 (age: 17.3 minutes)
```

**Solution:**
- Verify internet connection
- Check Alpaca API status
- Restart the bot
- Data age should be < 3 minutes for reliable trading

### Problem: Bot Not Trading

**Check these:**
1. **Buy conditions** - Are all conditions being met?
   ```
   INFO BUY CONDITIONS: UPTREND [PASS] | GOOD ENTRY [FAIL] | RSI [PASS] → No Buy Signal
   ```
   In this case, GOOD_ENTRY is blocking trades.

2. **Disable strict conditions** for testing:
   ```bash
   python mover.py --no-websocket --no-good-entry --no-rsi
   ```

3. **Wait for timeout buy** (5 minutes) to ensure activity.

4. **Check market hours** - Crypto trades 24/7 but ensure Alpaca API is accessible.

---

## CONCLUSION

This trading bot implements a complete scalping strategy with:
- **Real-time data** via alpaca-trade-api library (1-2 minute freshness)
- **Simulated fills** for reliable paper trading (no stuck orders)
- **Timeout buy feature** ensures trading activity every 5 minutes
- **Toggleable buy conditions** for strategy optimization and testing
- **Market orders** for immediate fills (or configurable limit orders)
- **Professional risk management** (position sizing, stops, trailing)
- **Advanced exit strategies** (partial profits, trailing stops)
- **Comprehensive performance tracking**
- **All indicators** specified in design document
- **Flexible configuration** via command-line arguments and code constants

### Key Improvements from Original Design

1. **Data Quality**: Switched to alpaca-trade-api for reliable real-time data (1-2 min vs 17+ hours)
2. **Order Execution**: Simulated fills solve paper trading order issues
3. **Timeout Safety**: Bot trades every 5 minutes minimum (prevents idle state)
4. **Testability**: Added condition toggles to A/B test strategy components
5. **Visibility**: Enhanced logging shows exact state of all conditions
6. **Profitability**: Adjusted fees/targets for viable net profit margins

### Recommended Usage

For testing and development:
```bash
python mover.py --mode paper --no-websocket
```

With all conditions disabled (fast testing):
```bash
python mover.py --mode paper --no-websocket --no-uptrend --no-good-entry --no-rsi
```

For production trading (when ready):
```bash
python mover.py --mode live --no-websocket
```

**Important Notes:**
- Always verify data freshness in logs (< 3 minutes)
- Monitor timeout buys - they trade regardless of conditions
- Use simulated fills for paper trading (SIMULATE_FILLS = True)
- Clean up stuck orders with cancel_all_orders.py if needed
- Test thoroughly in paper mode before live trading!
