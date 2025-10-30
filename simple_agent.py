#!/usr/bin/env python3
"""
Simple Crypto Agent with WebSocket Server
Monitors multiple cryptocurrencies for SMA crossover patterns and broadcasts via WebSocket.

Author: AI Assistant
Date: 2025-10-30
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

import pytz
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Alpaca imports
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
SYMBOLS = ["BTC/USD", "DOGE/USD", "ETH/USD"]  # Multiple crypto symbols
SHORT_WINDOW = 50   # Short-term SMA
LONG_WINDOW = 200   # Long-term SMA
CHECK_INTERVAL = 60  # Check every minute for demo purposes
ALERT_COOLDOWN = 300  # 5 minutes between same pattern alerts

# Validate API credentials
if not API_KEY or not API_SECRET:
    raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env file")

# Initialize Alpaca client
alpaca_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class CrossoverAlert:
    """Data class for crossover alerts"""
    timestamp: datetime
    symbol: str
    pattern: str  # "Golden Cross" or "Death Cross"
    price: float
    sma_short: float
    sma_long: float
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "pattern": self.pattern,
            "price": self.price,
            "sma_short": self.sma_short,
            "sma_long": self.sma_long
        }

@dataclass
class MarketData:
    """Data class for current market data"""
    symbol: str
    price: float
    sma_short: float
    sma_long: float
    trend: str  # "bullish", "bearish", "neutral"
    last_updated: datetime
    ai_analysis: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "sma_short": self.sma_short,
            "sma_long": self.sma_long,
            "trend": self.trend,
            "last_updated": self.last_updated.isoformat(),
            "ai_analysis": self.ai_analysis
        }

class WebSocketManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            logger.debug("No active WebSocket connections to broadcast to")
            return

        # Serialize message and compute diagnostics
        try:
            message_text = json.dumps(message, default=str)
        except Exception:
            # Fallback: stringify cheaply
            message_text = str(message)

        disconnected = set()

        # Log a concise preview of what we're sending
        try:
            msg_type = message.get("type") if isinstance(message, dict) else None
            keys = list(message.keys()) if isinstance(message, dict) else []
        except Exception:
            msg_type = None
            keys = []

        size = len(message_text)
        preview = (message_text[:400] + "...") if size > 400 else message_text
        logger.info(f"WebSocket Broadcast -> Type: {msg_type} | Keys: {keys} | Clients: {len(self.active_connections)} | Size: {size} bytes")
        logger.debug(f"Payload preview: {preview}")

        # Send to each connection and remove dead ones
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        for connection in disconnected:
            try:
                self.active_connections.discard(connection)
            except Exception:
                pass

class CryptoAgentWebSocket:
    """Simple crypto agent with WebSocket broadcasting"""
    
    def __init__(self):
        self.last_alerts: Dict[str, float] = {}  # Track alert cooldowns
        self.alerts_log: List[CrossoverAlert] = []
        self.market_data: Dict[str, MarketData] = {}
        self.websocket_manager = WebSocketManager()
        
    def get_crypto_data(self, symbol: str, limit: int = 300) -> pd.DataFrame:
        """
        Fetch historical price data for a cryptocurrency symbol.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC/USD")
            limit: Number of data points to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(days=15)  # Get enough data for 200-period SMA
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,  # Use hourly data for better signal quality
                start=start_time,
                end=end_time,
                limit=limit
            )
            
            bars = alpaca_client.get_crypto_bars(request).df
            
            if bars.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            # Filter for specific symbol and reset index
            df = bars[bars.index.get_level_values("symbol") == symbol].copy()
            df = df.reset_index()
            
            # Ensure numeric types
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            logger.debug(f"Fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_sma(self, df: pd.DataFrame, window: int, column: str = "close") -> pd.Series:
        """Calculate Simple Moving Average"""
        return df[column].rolling(window=window, min_periods=window).mean()
    
    async def get_ai_analysis(self, symbol: str, price: float, sma_short: float, sma_long: float, trend: str, recent_alerts: List[CrossoverAlert]) -> str:
        """
        Generate short AI market analysis for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol
            price: Current price
            sma_short: Short-term SMA
            sma_long: Long-term SMA
            trend: Current trend (bullish/bearish/neutral)
            recent_alerts: Recent alerts for this symbol
            
        Returns:
            Short AI analysis string
        """
        try:
            # Count recent patterns
            recent_golden_crosses = len([a for a in recent_alerts if a.pattern == "Golden Cross" and a.symbol == symbol])
            recent_death_crosses = len([a for a in recent_alerts if a.pattern == "Death Cross" and a.symbol == symbol])
            
            # Calculate SMA relationship
            sma_diff_percent = ((sma_short - sma_long) / sma_long * 100) if sma_long > 0 else 0
            
            # Build context for AI
            context = f"""Analyze {symbol} cryptocurrency briefly:

Current Price: ${price:.2f}
SMA 50: ${sma_short:.2f}
SMA 200: ${sma_long:.2f}
Current Trend: {trend}
SMA Difference: {sma_diff_percent:+.2f}%

Recent Patterns (last hour):
- Golden Crosses: {recent_golden_crosses}
- Death Crosses: {recent_death_crosses}

Provide a very brief trading insight (1-2 sentences only) covering:
1. Current market sentiment
2. Short-term outlook
Keep it concise and actionable."""

            # Generate simple technical analysis summary
            analysis_parts = []
            
            # Analyze SMA relationship
            if sma_diff_percent > 2:
                analysis_parts.append(f"{symbol} shows bullish structure with short MA {sma_diff_percent:.1f}% above long MA.")
            elif sma_diff_percent < -2:
                analysis_parts.append(f"{symbol} in bearish structure with short MA {abs(sma_diff_percent):.1f}% below long MA.")
            else:
                analysis_parts.append(f"{symbol} showing neutral MA alignment at ${price:.2f}.")
            
            # Add crossover context
            if recent_golden_crosses > 0:
                analysis_parts.append("Recent golden cross signals suggest upward momentum.")
            elif recent_death_crosses > 0:
                analysis_parts.append("Death cross patterns indicate potential downside risk.")
            else:
                analysis_parts.append("Monitor for MA crossover signals.")
            
            analysis = " ".join(analysis_parts)
            
            logger.info(f"AI Analysis for {symbol}: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis error for {symbol}: {e}")
            return f"AI analysis temporarily unavailable for {symbol}"
    
    def detect_crossover(self, df: pd.DataFrame, symbol: str) -> Optional[CrossoverAlert]:
        """
        Detect Golden Cross or Death Cross patterns.
        
        Args:
            df: Price data DataFrame
            symbol: Crypto symbol
            
        Returns:
            CrossoverAlert if pattern detected, None otherwise
        """
        if len(df) < LONG_WINDOW + 1:
            logger.debug(f"Insufficient data for {symbol}: {len(df)} points")
            return None
        
        # Calculate moving averages
        df["sma_short"] = self.calculate_sma(df, SHORT_WINDOW)
        df["sma_long"] = self.calculate_sma(df, LONG_WINDOW)
        
        # Get latest values (need at least 2 points for crossover detection)
        if len(df) < 2:
            return None
            
        # Current and previous values
        curr_short = df["sma_short"].iloc[-1]
        curr_long = df["sma_long"].iloc[-1]
        prev_short = df["sma_short"].iloc[-2]
        prev_long = df["sma_long"].iloc[-2]
        curr_price = df["close"].iloc[-1]
        
        # Check for valid data
        if pd.isna(curr_short) or pd.isna(curr_long) or pd.isna(prev_short) or pd.isna(prev_long):
            return None
        
        # Update market data
        trend = "neutral"
        if curr_short > curr_long:
            trend = "bullish"
        elif curr_short < curr_long:
            trend = "bearish"
        
        # Update market data (AI analysis will be added separately)
        existing_analysis = None
        if symbol in self.market_data:
            existing_analysis = self.market_data[symbol].ai_analysis
        
        self.market_data[symbol] = MarketData(
            symbol=symbol,
            price=float(curr_price),
            sma_short=float(curr_short),
            sma_long=float(curr_long),
            trend=trend,
            last_updated=datetime.now(pytz.UTC),
            ai_analysis=existing_analysis  # Keep existing analysis until updated
        )
        
        # Detect crossovers
        pattern = None
        if prev_short <= prev_long and curr_short > curr_long:
            pattern = "Golden Cross"
        elif prev_short >= prev_long and curr_short < curr_long:
            pattern = "Death Cross"
        
        if pattern:
            return CrossoverAlert(
                timestamp=datetime.now(pytz.UTC),
                symbol=symbol,
                pattern=pattern,
                price=float(curr_price),
                sma_short=float(curr_short),
                sma_long=float(curr_long)
            )
        
        return None
    
    def should_send_alert(self, symbol: str, pattern: str) -> bool:
        """
        Check if we should send an alert based on cooldown period.
        
        Args:
            symbol: Crypto symbol
            pattern: Pattern type
            
        Returns:
            True if alert should be sent
        """
        alert_key = f"{symbol}_{pattern}"
        now = datetime.now().timestamp()
        
        last_alert_time = self.last_alerts.get(alert_key, 0)
        
        if now - last_alert_time >= ALERT_COOLDOWN:
            self.last_alerts[alert_key] = now
            return True
        
        return False
    
    async def send_alert(self, alert: CrossoverAlert) -> None:
        """
        Send alert to console, log file, and WebSocket clients.
        
        Args:
            alert: CrossoverAlert instance
        """
        # Format alert message
        timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"{alert.symbol}: {alert.pattern} detected! "
            f"Price: ${alert.price:.2f}, "
            f"SMA_{SHORT_WINDOW}: ${alert.sma_short:.2f}, "
            f"SMA_{LONG_WINDOW}: ${alert.sma_long:.2f}"
        )
        
        # Print to console with color
        if alert.pattern == "Golden Cross":
            print(f"\033[92m[{timestamp_str}] 🚀 {message}\033[0m")  # Green
        else:
            print(f"\033[91m[{timestamp_str}] 📉 {message}\033[0m")  # Red
        
        # Log to file
        logger.info(message)
        
        # Store in memory
        self.alerts_log.append(alert)
        
        # Broadcast to WebSocket clients
        await self.websocket_manager.broadcast({
            "type": "alert",
            "data": alert.to_dict()
        })
        
        # Save to JSON file
        self.save_alerts_to_file()
    
    def save_alerts_to_file(self) -> None:
        """Save alerts to JSON file"""
        try:
            alerts_data = []
            for alert in self.alerts_log[-100:]:  # Keep last 100 alerts
                alerts_data.append(alert.to_dict())
            
            with open("crypto_alerts.json", "w") as f:
                json.dump(alerts_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alerts to file: {e}")
    
    async def monitor_symbol(self, symbol: str) -> None:
        """
        Monitor a single cryptocurrency symbol.
        
        Args:
            symbol: Crypto symbol to monitor
        """
        try:
            logger.debug(f"Checking {symbol}...")
            
            # Fetch data
            df = self.get_crypto_data(symbol)
            if df.empty:
                return
            
            # Detect crossover
            alert = self.detect_crossover(df, symbol)
            
            if alert and self.should_send_alert(symbol, alert.pattern):
                await self.send_alert(alert)
            
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")
    
    async def update_ai_analysis(self):
        """Update AI analysis for all symbols"""
        try:
            for symbol, market_data in self.market_data.items():
                # Get recent alerts for this symbol
                recent_alerts = [a for a in self.alerts_log if 
                               a.symbol == symbol and 
                               (datetime.now(pytz.UTC) - a.timestamp).total_seconds() < 3600]  # Last hour
                
                # Get AI analysis
                ai_analysis = await self.get_ai_analysis(
                    symbol, market_data.price, market_data.sma_short, 
                    market_data.sma_long, market_data.trend, recent_alerts
                )
                
                # Update the market data with AI analysis
                self.market_data[symbol].ai_analysis = ai_analysis
                
        except Exception as e:
            logger.error(f"Error updating AI analysis: {e}")
    
    async def broadcast_market_data(self):
        """Broadcast current market data to all clients"""
        market_data_dict = {
            symbol: data.to_dict() 
            for symbol, data in self.market_data.items()
        }
        
        await self.websocket_manager.broadcast({
            "type": "market_data",
            "data": market_data_dict,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })
    
    async def run_monitoring_loop(self) -> None:
        """
        Main monitoring loop - continuously check all symbols.
        """
        logger.info("🤖 Crypto Agent WebSocket Server Started!")
        logger.info(f"Monitoring symbols: {', '.join(SYMBOLS)}")
        logger.info(f"SMA windows: {SHORT_WINDOW} / {LONG_WINDOW}")
        logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
        logger.info("=" * 60)
        
        iteration = 0
        
        while True:
            iteration += 1
            start_time = datetime.now()
            
            logger.info(f"🔍 Scan #{iteration} - {start_time.strftime('%H:%M:%S')}")
            
            # Monitor all symbols concurrently
            tasks = [self.monitor_symbol(symbol) for symbol in SYMBOLS]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update AI analysis every few iterations to avoid overloading
            if iteration % 3 == 0:  # Update AI analysis every 3rd iteration
                logger.info("🧠 Updating AI analysis...")
                await self.update_ai_analysis()
            
            # Broadcast current market data
            await self.broadcast_market_data()
            
            # Calculate scan duration
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Scan completed in {duration:.2f}s")
            
            # Display summary
            if self.alerts_log:
                recent_alerts = len([a for a in self.alerts_log if 
                                   (datetime.now(pytz.UTC) - a.timestamp).total_seconds() < 3600])
                logger.info(f"📊 Total alerts: {len(self.alerts_log)}, Recent (1h): {recent_alerts}")
            
            logger.info(f"⏰ Next scan in {CHECK_INTERVAL} seconds...")
            logger.info("-" * 40)
            
            # Wait for next iteration
            await asyncio.sleep(CHECK_INTERVAL)

# Initialize the crypto agent
crypto_agent = CryptoAgentWebSocket()

# FastAPI app
app = FastAPI(title="Crypto Agent WebSocket Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Start the monitoring loop when the app starts"""
    asyncio.create_task(crypto_agent.run_monitoring_loop())

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Crypto Agent WebSocket Server", "status": "running"}

@app.get("/status")
async def get_status():
    """Get current status"""
    return {
        "status": "running",
        "symbols": SYMBOLS,
        "sma_windows": {"short": SHORT_WINDOW, "long": LONG_WINDOW},
        "check_interval": CHECK_INTERVAL,
        "total_alerts": len(crypto_agent.alerts_log),
        "active_connections": len(crypto_agent.websocket_manager.active_connections),
        "market_data": {symbol: data.to_dict() for symbol, data in crypto_agent.market_data.items()}
    }

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    recent_alerts = crypto_agent.alerts_log[-limit:] if crypto_agent.alerts_log else []
    return {
        "alerts": [alert.to_dict() for alert in reversed(recent_alerts)],
        "total": len(crypto_agent.alerts_log)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await crypto_agent.websocket_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            
    except WebSocketDisconnect:
        crypto_agent.websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        crypto_agent.websocket_manager.disconnect(websocket)

if __name__ == "__main__":
    # Run the FastAPI app object directly instead of using an import string
    # This avoids uvicorn attempting to import a module named 'crypto_agent_websocket'
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)