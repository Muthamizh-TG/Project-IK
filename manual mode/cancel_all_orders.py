"""
Cancel all open orders for the trading bot.
Run this to clean up stuck orders from paper trading.
"""

import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# API credentials (same as mover.py)
API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
API_SECRET = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"
PAPER = True

def cancel_all_orders():
    """Cancel all open orders"""
    trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    
    # Get all open orders
    try:
        open_orders = trading_client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN
        ))
        
        if not open_orders:
            logging.info("No open orders to cancel")
            return
        
        logging.info(f"Found {len(open_orders)} open order(s)")
        
        # Cancel each order
        cancelled_count = 0
        failed_count = 0
        
        for order in open_orders:
            try:
                trading_client.cancel_order_by_id(order.id)
                logging.info(f"Cancelled order {order.id} - {order.symbol} {order.side} {order.qty} @ {order.order_type}")
                cancelled_count += 1
            except Exception as e:
                logging.error(f"Failed to cancel order {order.id}: {e}")
                failed_count += 1
        
        logging.info(f"\nSummary: Cancelled {cancelled_count} order(s), Failed {failed_count} order(s)")
        
    except Exception as e:
        logging.error(f"Error getting orders: {e}")

if __name__ == "__main__":
    logging.info("Starting order cancellation...")
    cancel_all_orders()
    logging.info("Done!")
