# ============================================================================
# MULTI-CRYPTO WEBSOCKET SERVER FOR FRONTEND
# ============================================================================
# This server broadcasts real-time data for multiple cryptocurrencies
# to the frontend React application
# ============================================================================

import asyncio
import json
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import TimeFrame
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
ALPACA_API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
ALPACA_SECRET_KEY = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"

# Cryptocurrencies to track with their display colors
CRYPTO_SYMBOLS = {
    'BTC/USD': {'color': '#F7931A', 'name': 'Bitcoin'},
    'ETH/USD': {'color': '#627EEA', 'name': 'Ethereum'},
    'SOL/USD': {'color': '#14F195', 'name': 'Solana'},
    'XRP/USD': {'color': '#2196f3', 'name': 'Ripple'},
    'DOGE/USD': {'color': '#C2A633', 'name': 'Dogecoin'},
    'HYPER/USD': {'color': '#FF6B6B', 'name': 'Hyper'},
    'PEPENODE/USD': {'color': '#4ECDC4', 'name': 'PepeNode'},
    'BEST/USD': {'color': '#95E1D3', 'name': 'Best'},
    'LILPEPE/USD': {'color': '#F38181', 'name': 'LilPepe'},
    'BONK/USD': {'color': '#AA96DA', 'name': 'Bonk'},
}

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================
app = FastAPI(title="Multi-Crypto WebSocket Server")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logging.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# ============================================================================
# CRYPTO DATA FETCHER
# ============================================================================
class MultiCryptoFetcher:
    def __init__(self):
        self.api = alpaca.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url="https://paper-api.alpaca.markets"
        )
        self.last_bars = {}
        logging.info("Multi-Crypto Fetcher initialized")

    async def fetch_all_crypto_data(self, limit=2):
        """Fetch latest data for all cryptocurrencies
        
        Args:
            limit: Number of bars to fetch (default 2 for latest bar + previous)
        """
        data = {}
        
        for symbol, config in CRYPTO_SYMBOLS.items():
            try:
                # Fetch bars (1-minute data)
                bars = self.api.get_crypto_bars(
                    symbol, 
                    TimeFrame.Minute,
                    limit=limit  # Get last N bars
                )
                
                bars_df = getattr(bars, 'df', bars)
                
                if len(bars_df) > 0:
                    latest_bar = bars_df.iloc[-1]
                    
                    # Calculate price change
                    price_change = 0
                    if len(bars_df) > 1:
                        prev_close = float(bars_df.iloc[-2]['close'])
                        curr_close = float(latest_bar['close'])
                        price_change = ((curr_close - prev_close) / prev_close) * 100
                    
                    data[symbol] = {
                        'symbol': symbol,
                        'name': config['name'],
                        'color': config['color'],
                        'price': float(latest_bar['close']),
                        'open': float(latest_bar['open']),
                        'high': float(latest_bar['high']),
                        'low': float(latest_bar['low']),
                        'close': float(latest_bar['close']),
                        'volume': float(latest_bar['volume']),
                        'timestamp': latest_bar.name.isoformat() if hasattr(latest_bar.name, 'isoformat') else str(latest_bar.name),
                        'change': price_change,
                        'available': True,
                        # Add bar data in format similar to LiveChart
                        'bar': {
                            'o': float(latest_bar['open']),
                            'h': float(latest_bar['high']),
                            'l': float(latest_bar['low']),
                            'c': float(latest_bar['close']),
                            'v': float(latest_bar['volume']),
                            't': latest_bar.name.isoformat() if hasattr(latest_bar.name, 'isoformat') else str(latest_bar.name)
                        }
                    }
                else:
                    logging.warning(f"{symbol} returned no data - may not be available on Alpaca")
                    
            except Exception as e:
                # Check if it's a "symbol not found" error
                error_msg = str(e).lower()
                if 'not found' in error_msg or 'invalid' in error_msg or 'does not exist' in error_msg:
                    logging.warning(f"⚠️  {symbol} is not available on Alpaca Markets")
                else:
                    logging.error(f"Error fetching {symbol}: {e}")
                
                # Send last known data if available
                if symbol in self.last_bars:
                    data[symbol] = self.last_bars[symbol]
        
        # Update last bars cache
        self.last_bars.update(data)
        return data

    async def fetch_historical_crypto_data(self, minutes=15):
        """Fetch historical data for all cryptocurrencies (last N minutes)
        
        Args:
            minutes: Number of minutes of historical data to fetch
        """
        data = {}
        
        for symbol, config in CRYPTO_SYMBOLS.items():
            try:
                # Fetch historical bars (1-minute data) for the past N minutes
                bars = self.api.get_crypto_bars(
                    symbol, 
                    TimeFrame.Minute,
                    limit=minutes + 1  # Get one extra to calculate price change
                )
                
                bars_df = getattr(bars, 'df', bars)
                
                if len(bars_df) > 0:
                    # Return all bars as a list of data points
                    bars_list = []
                    for idx, bar in bars_df.iterrows():
                        bars_list.append({
                            'time': idx.strftime('%H:%M:%S') if hasattr(idx, 'strftime') else str(idx),
                            'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                            'price': float(bar['close']),
                            'open': float(bar['open']),
                            'high': float(bar['high']),
                            'low': float(bar['low']),
                            'close': float(bar['close']),
                            'volume': float(bar['volume']),
                        })
                    
                    # Calculate price change from first to last bar
                    price_change = 0
                    if len(bars_df) > 1:
                        first_close = float(bars_df.iloc[0]['close'])
                        last_close = float(bars_df.iloc[-1]['close'])
                        price_change = ((last_close - first_close) / first_close) * 100
                    
                    data[symbol] = {
                        'symbol': symbol,
                        'name': config['name'],
                        'color': config['color'],
                        'bars': bars_list,  # Historical bars data
                        'price': float(bars_df.iloc[-1]['close']),
                        'open': float(bars_df.iloc[-1]['open']),
                        'high': float(bars_df.iloc[-1]['high']),
                        'low': float(bars_df.iloc[-1]['low']),
                        'close': float(bars_df.iloc[-1]['close']),
                        'volume': float(bars_df.iloc[-1]['volume']),
                        'change': price_change,
                        'available': True,
                    }
                else:
                    logging.warning(f"{symbol} returned no historical data")
                    
            except Exception as e:
                # Check if it's a "symbol not found" error
                error_msg = str(e).lower()
                if 'not found' in error_msg or 'invalid' in error_msg or 'does not exist' in error_msg:
                    logging.warning(f"⚠️  {symbol} is not available on Alpaca Markets")
                else:
                    logging.error(f"Error fetching historical data for {symbol}: {e}")
        
        return data

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    fetcher = MultiCryptoFetcher()
    
    try:
        # On connection, send past 15 minutes of historical data
        logging.info("New client connected, sending historical data...")
        historical_data = await fetcher.fetch_historical_crypto_data(minutes=15)
        
        if historical_data:
            historical_message = {
                'type': 'historical_data',
                'timestamp': datetime.utcnow().isoformat(),
                'data': historical_data
            }
            await websocket.send_json(historical_message)
            logging.info(f"Sent historical data for {len(historical_data)} cryptos")
        
        # Keep connection alive
        while True:
            # Wait for any message from client (ping/pong)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# BACKGROUND TASK: FETCH AND BROADCAST DATA
# ============================================================================
async def broadcast_crypto_data():
    """Background task that fetches and broadcasts crypto data every 5 minutes"""
    fetcher = MultiCryptoFetcher()
    
    while True:
        try:
            # Fetch data for all cryptos
            crypto_data = await fetcher.fetch_all_crypto_data()
            
            if crypto_data:
                # Prepare message for frontend
                message = {
                    'type': 'multi_crypto_update',
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': crypto_data
                }
                
                # Broadcast to all connected clients
                await manager.broadcast(message)
                
                logging.info(f"Broadcasted data for {len(crypto_data)} cryptos to {len(manager.active_connections)} clients")
            
        except Exception as e:
            logging.error(f"Error in broadcast loop: {e}")
        
        # Wait 300 seconds (5 minutes) before next update
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    """Start the background broadcast task when server starts"""
    asyncio.create_task(broadcast_crypto_data())
    logging.info("Multi-Crypto WebSocket Server started")
    logging.info(f"Attempting to track {len(CRYPTO_SYMBOLS)} cryptocurrencies")
    logging.info("Note: Some tokens may not be available on Alpaca Markets")
    logging.info("📊 Data Update Frequency:")
    logging.info("   - On connection: Past 15 minutes of historical data")
    logging.info("   - Continuous: New data every 5 minutes")

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Multi-Crypto WebSocket Server",
        "symbols": list(CRYPTO_SYMBOLS.keys()),
        "endpoint": "/ws"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "tracking_symbols": len(CRYPTO_SYMBOLS)
    }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("MULTI-CRYPTO WEBSOCKET SERVER")
    print("=" * 60)
    print(f"Tracking {len(CRYPTO_SYMBOLS)} cryptocurrencies:")
    for symbol, config in CRYPTO_SYMBOLS.items():
        print(f"  - {config['name']} ({symbol})")
    print("\n📊 DATA UPDATE FREQUENCY:")
    print("   🔄 On connection: Past 15 minutes of historical data")
    print("   ⏰ Continuous: New data every 5 minutes (300 seconds)")
    print("\n⚠️  NOTE: Some tokens may not be available on Alpaca")
    print("   Available: BTC, ETH, SOL, XRP, DOGE (confirmed)")
    print("   Check logs for unavailable tokens")
    print("\nServer starting on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
