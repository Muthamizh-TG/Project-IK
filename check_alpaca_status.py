"""
Check Current Alpaca Paper Trading Status
Shows: Positions, Open Orders, Recent Orders
"""
import alpaca_trade_api as alpaca

# Your Alpaca credentials
ALPACA_API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
ALPACA_SECRET_KEY = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"

# Initialize API
api = alpaca.REST(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    base_url="https://paper-api.alpaca.markets"
)

print("\n" + "="*70)
print("   ALPACA PAPER TRADING STATUS CHECK")
print("="*70)

# 1. Check Account
print("\n[ACCOUNT INFO]:")
try:
    account = api.get_account()
    print(f"   Status: {account.status}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Cash: ${float(account.cash):,.2f}")
    print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
except Exception as e:
    print(f"   Error: {e}")

# 2. Check Current Positions
print("\n[CURRENT POSITIONS]:")
try:
    positions = api.list_positions()
    if positions:
        for pos in positions:
            print(f"\n   Symbol: {pos.symbol}")
            print(f"   Quantity: {pos.qty}")
            print(f"   Entry Price: ${float(pos.avg_entry_price):.4f}")
            print(f"   Current Price: ${float(pos.current_price):.4f}")
            print(f"   P&L: ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc)*100:.2f}%)")
    else:
        print("   No open positions")
except Exception as e:
    print(f"   Error: {e}")

# 3. Check Open Orders
print("\n[OPEN ORDERS]:")
try:
    open_orders = api.list_orders(status='open')
    if open_orders:
        for order in open_orders:
            print(f"\n   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side.upper()}")
            print(f"   Type: {order.type.upper()}")
            print(f"   Quantity: {order.qty}")
            print(f"   Status: {order.status.upper()}")
            print(f"   Created: {order.created_at}")
    else:
        print("   No open orders")
except Exception as e:
    print(f"   Error: {e}")

# 4. Check Recent Closed Orders (last 10)
print("\n[RECENT CLOSED ORDERS] (last 10):")
try:
    closed_orders = api.list_orders(status='closed', limit=10)
    if closed_orders:
        for order in closed_orders:
            print(f"\n   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side.upper()}")
            print(f"   Quantity: {order.qty}")
            if order.filled_qty:
                print(f"   Filled: {order.filled_qty} @ ${order.filled_avg_price}")
            print(f"   Status: {order.status.upper()}")
            print(f"   Created: {order.created_at}")
    else:
        print("   No recent closed orders")
except Exception as e:
    print(f"   Error: {e}")

# 5. Check All Orders (last 20)
print("\n[ALL RECENT ORDERS] (last 20):")
try:
    all_orders = api.list_orders(status='all', limit=20)
    print(f"\n   Total orders found: {len(all_orders)}")
    
    # Count by status
    status_count = {}
    for order in all_orders:
        status = order.status
        status_count[status] = status_count.get(status, 0) + 1
    
    print("\n   Orders by status:")
    for status, count in status_count.items():
        print(f"      {status.upper()}: {count}")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
print("\nStatus check complete!\n")
