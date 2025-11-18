#!/usr/bin/env python3
"""
Close Position Script for Alpaca Paper Trading
Closes open positions via market orders.
"""

import argparse
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Alpaca API credentials (paper trading)
API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
API_SECRET = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"
PAPER = True

def get_alpaca_client():
    """Initialize and return Alpaca trading client."""
    return TradingClient(API_KEY, API_SECRET, paper=PAPER)

def close_position(symbol: str, api: TradingClient = None):
    """
    Close a specific position by symbol.
    
    Args:
        symbol: The symbol to close (e.g., 'XRP/USD' or 'XRPUSD')
        api: Optional TradingClient instance
        
    Returns:
        True if successful, False otherwise
    """
    if api is None:
        api = get_alpaca_client()
    
    # Normalize symbol format (Alpaca uses 'XRP/USD' format for crypto)
    if 'USD' in symbol and '/' not in symbol:
        # Convert XRPUSD -> XRP/USD
        base = symbol.replace('USD', '')
        symbol = f"{base}/USD"
    
    try:
        # Get current position
        position = api.get_open_position(symbol)
        
        qty = float(position.qty)
        side = position.side
        current_price = float(position.current_price)
        market_value = float(position.market_value)
        unrealized_pl = float(position.unrealized_pl)
        unrealized_plpc = float(position.unrealized_plpc)
        
        print(f"\n{'='*70}")
        print(f"POSITION FOUND: {symbol}")
        print(f"{'='*70}")
        print(f"  Quantity:        {qty:,.8f}")
        print(f"  Side:            {side}")
        print(f"  Current Price:   ${current_price:,.5f}")
        print(f"  Market Value:    ${market_value:,.2f}")
        print(f"  Unrealized P/L:  ${unrealized_pl:,.2f} ({unrealized_plpc*100:+.2f}%)")
        print(f"{'='*70}\n")
        
        # Determine order side (opposite of position side)
        order_side = OrderSide.SELL if side == 'long' else OrderSide.BUY
        
        # Create market order to close position
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=order_side,
            time_in_force=TimeInForce.GTC
        )
        
        print(f"Submitting {order_side.value} order to close position...")
        
        # Submit order
        order = api.submit_order(order_data)
        
        print(f"✓ Order submitted successfully!")
        print(f"  Order ID:   {order.id}")
        print(f"  Status:     {order.status}")
        print(f"  Side:       {order.side}")
        print(f"  Quantity:   {order.qty}")
        print(f"  Type:       {order.type}")
        print(f"\nNote: In Alpaca paper crypto, SELL orders typically fill immediately,")
        print(f"      but BUY orders may stay in 'new' status indefinitely.\n")
        
        return True
        
    except Exception as e:
        if "position does not exist" in str(e).lower() or "not found" in str(e).lower():
            print(f"\n✗ No open position found for {symbol}")
        else:
            print(f"\n✗ Error closing position for {symbol}: {e}")
        return False

def close_all_positions(api: TradingClient = None):
    """
    Close all open positions.
    
    Args:
        api: Optional TradingClient instance
        
    Returns:
        Number of positions closed
    """
    if api is None:
        api = get_alpaca_client()
    
    try:
        # Get all positions
        positions = api.get_all_positions()
        
        if not positions:
            print("\nNo open positions to close.")
            return 0
        
        print(f"\n{'='*70}")
        print(f"FOUND {len(positions)} OPEN POSITION(S)")
        print(f"{'='*70}\n")
        
        closed_count = 0
        for position in positions:
            # Use the exact symbol from the position (don't convert it)
            symbol = position.symbol
            
            # Get position details
            qty = float(position.qty)
            side = position.side
            current_price = float(position.current_price)
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc)
            
            print(f"{'='*70}")
            print(f"POSITION: {symbol}")
            print(f"{'='*70}")
            print(f"  Quantity:        {qty:,.8f}")
            print(f"  Side:            {side}")
            print(f"  Current Price:   ${current_price:,.5f}")
            print(f"  Market Value:    ${market_value:,.2f}")
            print(f"  Unrealized P/L:  ${unrealized_pl:,.2f} ({unrealized_plpc*100:+.2f}%)")
            print(f"{'='*70}\n")
            
            # Determine order side (opposite of position side)
            order_side = OrderSide.SELL if str(side) == 'PositionSide.LONG' or 'long' in str(side).lower() else OrderSide.BUY
            
            # Create market order to close position
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=order_side,
                time_in_force=TimeInForce.GTC
            )
            
            print(f"Submitting {order_side.value} order to close {symbol}...")
            
            try:
                # Submit order
                order = api.submit_order(order_data)
                
                print(f"✓ Order submitted successfully!")
                print(f"  Order ID:   {order.id}")
                print(f"  Status:     {order.status}")
                print(f"  Side:       {order.side}")
                print(f"  Quantity:   {order.qty}")
                print(f"  Type:       {order.type}\n")
                
                closed_count += 1
                
            except Exception as e:
                print(f"✗ Error submitting order for {symbol}: {e}\n")
        
        print(f"{'='*70}")
        print(f"SUMMARY: Successfully submitted close orders for {closed_count}/{len(positions)} position(s)")
        print(f"{'='*70}\n")
        
        return closed_count
        
    except Exception as e:
        print(f"\n✗ Error fetching positions: {e}")
        return 0

def list_positions(api: TradingClient = None):
    """
    List all open positions without closing them.
    
    Args:
        api: Optional TradingClient instance
    """
    if api is None:
        api = get_alpaca_client()
    
    try:
        positions = api.get_all_positions()
        
        if not positions:
            print("\nNo open positions.")
            return
        
        print(f"\n{'='*70}")
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"{'='*70}\n")
        
        for i, pos in enumerate(positions, 1):
            qty = float(pos.qty)
            current_price = float(pos.current_price)
            market_value = float(pos.market_value)
            unrealized_pl = float(pos.unrealized_pl)
            unrealized_plpc = float(pos.unrealized_plpc)
            
            print(f"{i}. {pos.symbol}")
            print(f"   Quantity:        {qty:,.8f}")
            print(f"   Side:            {pos.side}")
            print(f"   Current Price:   ${current_price:,.5f}")
            print(f"   Market Value:    ${market_value:,.2f}")
            print(f"   Unrealized P/L:  ${unrealized_pl:,.2f} ({unrealized_plpc*100:+.2f}%)")
            print()
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ Error fetching positions: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Close positions in Alpaca paper trading account",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Close a specific position
  python close-posicition.py --symbol XRPUSD
  python close-posicition.py --symbol XRP/USD
  
  # Close all positions
  python close-posicition.py --all
  
  # List positions without closing
  python close-posicition.py --list
        """
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Symbol to close (e.g., XRPUSD or XRP/USD)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Close all open positions'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List open positions without closing'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.symbol or args.all or args.list):
        parser.print_help()
        sys.exit(1)
    
    # Initialize API client
    api = get_alpaca_client()
    
    print(f"\n{'='*70}")
    print(f"ALPACA PAPER TRADING - CLOSE POSITION")
    print(f"{'='*70}")
    print(f"Account Status: Checking connection...")
    
    try:
        account = api.get_account()
        print(f"✓ Connected to account: {account.id}")
        print(f"  Status:         {account.status}")
        print(f"  Buying Power:   ${float(account.buying_power):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    except Exception as e:
        print(f"\n✗ Failed to connect to Alpaca: {e}")
        sys.exit(1)
    
    # Execute command
    if args.list:
        list_positions(api)
    elif args.all:
        close_all_positions(api)
    elif args.symbol:
        close_position(args.symbol, api)
    
    print("Done!\n")

if __name__ == "__main__":
    main()
