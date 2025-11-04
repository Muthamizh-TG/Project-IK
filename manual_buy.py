"""
Manual Buy Script - Buys XRP at current market price or custom limit price
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

def manual_buy():
    """Place a manual buy order for XRP"""
    
    # Get current XRP price
    try:
        # Check if we already have a position
        try:
            position = api.get_position('XRPUSD')
            print(f"\nWARNING: You already have an open position!")
            print(f"   Current Position: {position.qty} XRP")
            print(f"   Entry Price: ${position.avg_entry_price}")
            print(f"   Current P&L: ${position.unrealized_pl}")
            
            proceed = input(f"\n❓ Continue with another buy? (yes/no): ")
            if proceed.lower() not in ['yes', 'y']:
                print("\nBuy cancelled.")
                return
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
        
        # Get investment amount
        print(f"\nHow much do you want to invest?")
        lot_input = input(f"   Enter dollar amount (default: $2000): ").strip()
        lot = float(lot_input) if lot_input else 2000.0
        
        # Calculate quantity
        quantity = round(lot / current_price, 8)
        total_cost = quantity * current_price
        
        print(f"\nOrder Summary:")
        print(f"   Investment: ${lot:.2f}")
        print(f"   XRP Price: ${current_price:.6f}")
        print(f"   Quantity: {quantity} XRP")
        print(f"   Total Cost: ${total_cost:.2f}")
        
        # Choose order type
        print(f"\nOrder Type:")
        print(f"   1. MARKET order (buy immediately at current price)")
        print(f"   2. LIMIT order (buy only at your specified price)")
        
        order_type_input = input(f"   Choose (1 or 2, default: 1): ").strip()
        order_type = '2' if order_type_input == '2' else '1'
        
        if order_type == '1':
            # Market Order
            order_params = {
                'symbol': 'XRPUSD',
                'side': 'buy',
                'type': 'market',
                'qty': quantity,
                'time_in_force': 'gtc'
            }
            print(f"\nWill buy {quantity} XRP at MARKET price (~${current_price:.6f})")
            
        else:
            # Limit Order
            limit_price_input = input(f"\nEnter limit price (default: ${current_price:.6f}): ").strip()
            limit_price = float(limit_price_input) if limit_price_input else current_price
            
            order_params = {
                'symbol': 'XRPUSD',
                'side': 'buy',
                'type': 'limit',
                'qty': quantity,
                'limit_price': limit_price,
                'time_in_force': 'gtc'
            }
            print(f"\nWill buy {quantity} XRP at LIMIT price ${limit_price:.6f}")
            print(f"   (Order will only fill if price drops to ${limit_price:.6f} or below)")
        
        # Final confirmation
        confirm = input(f"\nPlace this buy order? (yes/no): ")

        if confirm.lower() in ['yes', 'y']:
            # Place the order
            order = api.submit_order(**order_params)

            print(f"\nBUY ORDER PLACED!")
            print(f"   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side.upper()}")
            print(f"   Type: {order.type.upper()}")
            print(f"   Quantity: {order.qty} XRP")
            if order.type == 'limit':
                print(f"   Limit Price: ${order.limit_price}")
            print(f"   Status: {order.status}")
            print(f"   Time in Force: {order.time_in_force.upper()}")
            
            if order.type == 'market':
                print(f"\nYour order should fill immediately!")
            else:
                print(f"\nYour order will fill when XRP price reaches ${order.limit_price}")
                print(f"   Current price: ${current_price:.6f}")
            
        else:
            print("\nBuy cancelled.")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nPossible reasons:")
        print("  - Insufficient buying power")
        print("  - API connection issue")
        print("  - Invalid quantity or price")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("   MANUAL BUY SCRIPT - XRP/USD")
    print("="*60)
    manual_buy()
    print("\n" + "="*60)
