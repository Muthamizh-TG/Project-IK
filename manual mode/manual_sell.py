"""
Manual Sell Script - Closes your current XRP position immediately
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

# Get current position
try:
    position = api.get_position('XRPUSD')
    print(f"\nCurrent Position:")
    print(f"   Symbol: {position.symbol}")
    print(f"   Quantity: {position.qty}")
    print(f"   Entry Price: ${position.avg_entry_price}")
    print(f"   Current Price: ${position.current_price}")
    print(f"   P&L: ${position.unrealized_pl} ({position.unrealized_plpc}%)")
    
    # Ask for confirmation
    confirm = input(f"\nSell {position.qty} XRP at market price? (yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        # Place market sell order (sells immediately)
        order = api.submit_order(
            symbol='XRPUSD',
            side='sell',
            type='market',
            qty=position.qty,
            time_in_force='gtc'
        )
        print(f"\nSELL ORDER PLACED!")
        print(f"   Order ID: {order.id}")
        print(f"   Status: {order.status}")
        print(f"   Quantity: {order.qty}")
        print(f"\nPosition will be closed at market price.")
    else:
        print("\nSale cancelled.")
        
except Exception as e:
    print(f"\nError: {e}")
    print("\nPossible reasons:")
    print("  - No open position for XRPUSD")
    print("  - API connection issue")
