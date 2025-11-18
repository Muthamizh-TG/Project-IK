"""
Auto Manual Buy Script - Runs manual_buy.py with automatic inputs for testing
"""
import alpaca_trade_api as alpaca
import time

# Your Alpaca credentials
ALPACA_API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
ALPACA_SECRET_KEY = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"

# Initialize API
api = alpaca.REST(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    base_url="https://paper-api.alpaca.markets"
)

print("\n" + "="*60)
print("   AUTO MANUAL BUY SCRIPT - XRP/USD")
print("="*60)

try:
    # Check if we already have a position
    try:
        position = api.get_position('XRPUSD')
        print(f"\nWARNING: You already have an open position!")
        print(f"   Current Position: {position.qty} XRP")
        print(f"   Entry Price: ${position.avg_entry_price}")
        print(f"   Current P&L: ${position.unrealized_pl}")
    except:
        print("\nNo existing position found. Ready to buy.")
    
    # Get latest XRP price
    try:
        bars = api.get_crypto_bars('XRP/USD', '1Min', limit=1)
        bars_df = getattr(bars, 'df', bars)
        current_price = float(bars_df['close'].iloc[-1])
    except:
        trade = api.get_latest_trade('XRP/USD')
        current_price = float(trade.price)
    
    print(f"\nCurrent XRP Price: ${current_price:.6f}")
    
    # Use default values
    lot = 100.0  # $100 investment for testing
    quantity = round(lot / current_price, 8)
    total_cost = quantity * current_price
    
    print(f"\nOrder Summary:")
    print(f"   Investment: ${lot:.2f}")
    print(f"   XRP Price: ${current_price:.6f}")
    print(f"   Quantity: {quantity} XRP")
    print(f"   Total Cost: ${total_cost:.2f}")
    
    # =======================================================================
    # TEST 1: LIMIT ORDER (Like main.py)
    # Place limit order BELOW current price to see if it fills
    # =======================================================================
    limit_price = current_price * 0.999  # 0.1% below market (aggressive limit)
    
    order_params = {
        'symbol': 'XRPUSD',
        'side': 'buy',
        'type': 'limit',  # LIMIT ORDER like main.py
        'qty': quantity,
        'limit_price': limit_price,
        'time_in_force': 'gtc'
    }
    print(f"\n[TEST: LIMIT ORDER] (like main.py)")
    print(f"   Placing LIMIT order at ${limit_price:.6f}")
    print(f"   (${current_price:.6f} - ${limit_price:.6f} = ${current_price - limit_price:.6f} below market)")
    
    # Place the order
    order = api.submit_order(**order_params)
    
    print(f"\n[ORDER PLACED]")
    print(f"   Order ID: {order.id}")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side.upper()}")
    print(f"   Type: {order.type.upper()}")
    print(f"   Quantity: {order.qty} XRP")
    print(f"   Limit Price: ${limit_price:.6f}")
    print(f"   Status: {order.status}")
    print(f"   Time in Force: {order.time_in_force.upper()}")
    print(f"   Created: {order.created_at}")
    
    # =======================================================================
    # MONITORING FILL - TWO METHODS (Like main.py)
    # Method 1: Check order status
    # Method 2: Check if position appears (main.py method)
    # =======================================================================
    print(f"\n[MONITORING ORDER FILL - DUAL METHOD]")
    print(f"Method 1: Check order status (order.status)")
    print(f"Method 2: Check position exists (api.get_position) <- main.py uses this!")
    print(f"Checking every 5 seconds...")
    print(f"Press Ctrl+C to stop\n")
    
    check_count = 0
    while True:  # Wait forever until filled or Ctrl+C
        time.sleep(5)
        check_count += 1
        
        # METHOD 1: Get order status
        order_status = api.get_order(order.id)
        
        # METHOD 2: Check if position exists (main.py method)
        has_position = False
        position_qty = 0
        try:
            position = api.get_position('XRPUSD')
            has_position = True
            position_qty = float(position.qty)
        except:
            has_position = False
        
        print(f"   Check #{check_count} ({check_count*5}s):")
        print(f"      [METHOD 1] Order Status: {order_status.status} | Filled: {order_status.filled_qty}")
        print(f"      [METHOD 2] Position Exists: {has_position} | Qty: {position_qty}")
        
        if order_status.filled_avg_price:
            print(f"      Fill Price: ${order_status.filled_avg_price}")
        
        # Check if filled by EITHER method
        if order_status.status == 'filled':
            print(f"\n✅ ORDER FILLED (Method 1: Order Status)")
            print(f"   Filled {order_status.filled_qty} XRP at ${order_status.filled_avg_price}")
            break
        
        if has_position and position_qty > 0:
            print(f"\n✅ POSITION DETECTED (Method 2: main.py method)")
            print(f"   Position: {position_qty} XRP")
            print(f"   Entry Price: ${position.avg_entry_price}")
            print(f"   This is how main.py detects fills!")
            break
        
        # If canceled or rejected, break
        if order_status.status in ['canceled', 'rejected', 'expired']:
            print(f"\n❌ ORDER {order_status.status.upper()}")
            break

except KeyboardInterrupt:
    print(f"\n\n[INTERRUPTED BY USER]")
    try:
        order_status = api.get_order(order.id)
        print(f"   Final Status: {order_status.status}")
        print(f"   Time waited: {check_count*5} seconds ({check_count*5/60:.1f} minutes)")
        
        if order_status.status == 'new':
            print(f"\n❌ ORDER STILL STUCK IN 'NEW' STATUS")
            print(f"   This proves Alpaca paper crypto buy orders DON'T FILL!")
            
            # Ask if user wants to cancel
            print(f"\n   The order is still open (ID: {order.id})")
            print(f"   You can cancel it later with cancel_all_orders.py")
    except:
        pass

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("\n")
