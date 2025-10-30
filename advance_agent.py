# server.py
import os
import time
import asyncio
import json
from datetime import datetime, timedelta
from typing import Set, Dict, Any, List
from contextlib import asynccontextmanager
import pytz
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np

# Alpaca imports (alpaca-py)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load .env
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTC/USD")
SHORT_WINDOW = int(os.getenv("SHORT_WINDOW", "50"))
LONG_WINDOW = int(os.getenv("LONG_WINDOW", "200"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "300"))  # seconds

# MongoDB configuration
MONGODB_URL = "mongodb+srv://InternUser:Intern@technologygarage.lmhpjkh.mongodb.net/"
DATABASE_NAME = "crypto_trading"

if not API_KEY or not API_SECRET:
    raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env")

# Alpaca client
client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

# MongoDB client and database
mongodb_client = None
db = None

async def init_mongodb():
    """Initialize MongoDB connection"""
    global mongodb_client, db
    try:
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        db = mongodb_client[DATABASE_NAME]
        
        # Test connection
        await mongodb_client.admin.command('ping')
        print("Connected to MongoDB successfully")
        
        # Create collections if they don't exist
        collections = ['market_data', 'alerts', 'ai_analysis', 'patterns']
        existing_collections = await db.list_collection_names()
        
        for collection_name in collections:
            if collection_name not in existing_collections:
                await db.create_collection(collection_name)
                print(f"Created collection: {collection_name}")
        
        # Create indexes for better performance
        await db.market_data.create_index([("timestamp", -1), ("symbol", 1)])
        await db.alerts.create_index([("timestamp", -1), ("symbol", 1)])
        await db.ai_analysis.create_index([("timestamp", -1), ("symbol", 1)])
        await db.patterns.create_index([("timestamp", -1), ("symbol", 1), ("pattern_type", 1)])
        
        print("Database indexes created successfully")
        
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongodb():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("MongoDB connection closed")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize MongoDB and start agent loop
    await init_mongodb()
    task = asyncio.create_task(agent_loop())
    yield
    # Shutdown: Cancel the task and close MongoDB
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await close_mongodb()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connected websocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            print("No active WebSocket connections")
            return

        text = json.dumps(message, default=str)
        clients = list(self.active_connections)
        sent = 0
        removed = 0

        # Send individually so we can handle/remove failing sockets
        for ws in clients:
            try:
                await ws.send_text(text)
                sent += 1
            except Exception as e:
                # Remove the websocket so future broadcasts don't attempt it
                try:
                    self.active_connections.discard(ws)
                except Exception:
                    pass
                removed += 1
                print(f"Removed dead WebSocket connection during broadcast: {e}")

        print(f"Broadcast complete. Sent: {sent}, Removed: {removed}, Message size: {len(text)} bytes")

manager = ConnectionManager()

last_alerts: Dict[str, float] = {}

# ---------- MongoDB Data Storage Functions ----------
async def store_market_data(symbol: str, bar_data: dict, indicators: dict):
    """Store market data with technical indicators"""
    try:
        document = {
                "timestamp": datetime.now(pytz.UTC),
            "symbol": symbol,
            "price": bar_data.get("c", 0),
            "open": bar_data.get("o", 0),
            "high": bar_data.get("h", 0),
            "low": bar_data.get("l", 0),
            "close": bar_data.get("c", 0),
            "volume": bar_data.get("v", 0),
            "rsi": indicators.get("rsi"),
            "sma_short": indicators.get("sma_short"),
            "sma_long": indicators.get("sma_long"),
            "price_change": indicators.get("price_change", 0)
        }
        await db.market_data.insert_one(document)
    except Exception as e:
        print(f"Error storing market data: {e}")

async def store_alert(symbol: str, alert_type: str, message: str, metadata: dict = None):
    """Store trading alerts"""
    try:
        document = {
                "timestamp": datetime.now(pytz.UTC),
            "symbol": symbol,
            "alert_type": alert_type,
            "message": message,
            "metadata": metadata or {},
            "processed": False
        }
        result = await db.alerts.insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error storing alert: {e}")
        return None

async def store_ai_analysis(symbol: str, analysis: str, context_data: dict):
    """Store AI analysis results"""
    try:
        document = {
            "timestamp": datetime.now(pytz.UTC),
            "symbol": symbol,
            "analysis": analysis,
            "context": context_data,
            "model": "moondream:1.8b"
        }
        await db.ai_analysis.insert_one(document)
    except Exception as e:
        print(f"Error storing AI analysis: {e}")

async def store_pattern(symbol: str, pattern_type: str, pattern_data: dict, confidence: float = 0.0):
    """Store detected patterns"""
    try:
        document = {
            "timestamp": datetime.now(pytz.UTC),
            "symbol": symbol,
            "pattern_type": pattern_type,  # 'golden_cross', 'death_cross', 'shooting_star', etc.
            "pattern_data": pattern_data,
            "confidence": confidence,
            "verified": False  # Can be updated later based on market movement
        }
        await db.patterns.insert_one(document)
    except Exception as e:
        print(f"Error storing pattern: {e}")

# ---------- Historical Data Retrieval Functions ----------
async def get_historical_market_data(symbol: str, hours_back: int = 24, limit: int = 1000):
    """Get historical market data from MongoDB"""
    try:
        start_time = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        cursor = db.market_data.find({
            "symbol": symbol,
            "timestamp": {"$gte": start_time}
        }).sort("timestamp", -1).limit(limit)
        
        data = []
        async for doc in cursor:
            data.append(doc)
        return data
    except Exception as e:
        print(f"Error retrieving historical data: {e}")
        return []

async def get_recent_alerts(symbol: str, hours_back: int = 24):
    """Get recent alerts from MongoDB"""
    try:
        start_time = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
        cursor = db.alerts.find({
            "symbol": symbol,
            "timestamp": {"$gte": start_time}
        }).sort("timestamp", -1)
        
        alerts = []
        async for doc in cursor:
            alerts.append(doc)
        return alerts
    except Exception as e:
        print(f"Error retrieving alerts: {e}")
        return []

async def get_historical_patterns(symbol: str, pattern_type: str = None, days_back: int = 7):
    """Get historical patterns for comparison"""
    try:
        start_time = datetime.now(pytz.UTC) - timedelta(days=days_back)
        query = {
            "symbol": symbol,
            "timestamp": {"$gte": start_time}
        }
        if pattern_type:
            query["pattern_type"] = pattern_type
            
        cursor = db.patterns.find(query).sort("timestamp", -1)
        
        patterns = []
        async for doc in cursor:
            patterns.append(doc)
        return patterns
    except Exception as e:
        print(f"Error retrieving patterns: {e}")
        return []

async def get_pattern_success_rate(pattern_type: str, days_back: int = 30):
    """Calculate success rate of a pattern type"""
    try:
        start_time = datetime.now(pytz.UTC) - timedelta(days=days_back)
        total_patterns = await db.patterns.count_documents({
            "pattern_type": pattern_type,
            "timestamp": {"$gte": start_time}
        })
        
        successful_patterns = await db.patterns.count_documents({
            "pattern_type": pattern_type,
            "timestamp": {"$gte": start_time},
            "verified": True
        })
        
        if total_patterns > 0:
            return successful_patterns / total_patterns
        return 0.0
    except Exception as e:
        print(f"Error calculating pattern success rate: {e}")
        return 0.0

async def get_price_trend_analysis(symbol: str, hours_back: int = 24):
    """Analyze price trends from historical data"""
    try:
        historical_data = await get_historical_market_data(symbol, hours_back)
        if len(historical_data) < 2:
            return {"trend": "insufficient_data", "strength": 0.0}
        
        prices = [float(d["price"]) for d in reversed(historical_data)]
        
        # Calculate trend strength using linear regression
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]
        
        # Determine trend direction and strength
        price_range = max(prices) - min(prices)
        if price_range > 0:
            trend_strength = abs(slope) / (price_range / len(prices))
        else:
            trend_strength = 0.0
        
        if slope > 0:
            trend = "bullish"
        elif slope < 0:
            trend = "bearish"
        else:
            trend = "sideways"
        
        return {
            "trend": trend,
            "strength": min(trend_strength, 1.0),
            "slope": slope,
            "price_change_24h": prices[-1] - prices[0] if len(prices) > 0 else 0
        }
    except Exception as e:
        print(f"Error analyzing price trend: {e}")
        return {"trend": "error", "strength": 0.0}

# ---------- Indicator & pattern helpers (ported from your script) ----------
def get_crypto_data_sync(symbol: str, limit=300) -> pd.DataFrame:
    """
    Synchronous data fetch. We'll call this via asyncio.to_thread.
    """
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=10)

    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time,
        limit=limit
    )
    bars = client.get_crypto_bars(req).df
    if bars.empty:
        return pd.DataFrame()
    df = bars[bars.index.get_level_values("symbol") == symbol].copy()
    df = df.reset_index()
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def calculate_sma(df: pd.DataFrame, window: int, column: str = "close") -> pd.Series:
    return df[column].rolling(window=window).mean()

async def check_cross(df: pd.DataFrame) -> str:
    """Enhanced crossover detection with historical pattern analysis"""
    if len(df) < LONG_WINDOW:
        return None
    df["SMA_short"] = calculate_sma(df, SHORT_WINDOW)
    df["SMA_long"] = calculate_sma(df, LONG_WINDOW)
    if pd.isna(df["SMA_long"].iloc[-1]) or pd.isna(df["SMA_short"].iloc[-1]):
        return None
    
    prev_short, curr_short = df["SMA_short"].iloc[-2], df["SMA_short"].iloc[-1]
    prev_long, curr_long = df["SMA_long"].iloc[-2], df["SMA_long"].iloc[-1]
    curr_price = df["close"].iloc[-1]
    
    if prev_short < prev_long and curr_short > curr_long:
        # Golden Cross detected
        pattern_type = "golden_cross"
        success_rate = await get_pattern_success_rate(pattern_type, 30)
        
        # Store pattern in database
        pattern_data = {
            "price": float(curr_price),
            "sma_short": float(curr_short),
            "sma_long": float(curr_long),
            "volume": float(df["volume"].iloc[-1]) if "volume" in df else 0
        }
        await store_pattern(SYMBOL, pattern_type, pattern_data, success_rate)
        
        return f"Golden Cross — Potential uptrend (Success rate: {success_rate:.1%}). Price: {curr_price:.2f}"
        
    elif prev_short > prev_long and curr_short < curr_long:
        # Death Cross detected
        pattern_type = "death_cross"
        success_rate = await get_pattern_success_rate(pattern_type, 30)
        
        # Store pattern in database
        pattern_data = {
            "price": float(curr_price),
            "sma_short": float(curr_short),
            "sma_long": float(curr_long),
            "volume": float(df["volume"].iloc[-1]) if "volume" in df else 0
        }
        await store_pattern(SYMBOL, pattern_type, pattern_data, success_rate)
        
        return f"Death Cross — Potential downtrend (Success rate: {success_rate:.1%}). Price: {curr_price:.2f}"
    
    return None

# Candlestick patterns
def is_shooting_star(row: pd.Series) -> bool:
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['close'], row['open'])
    lower_shadow = min(row['close'], row['open']) - row['low']
    return (upper_shadow > 2 * body) and (lower_shadow < body * 0.5) and (row['close'] < row['open'])

def is_hammer(row: pd.Series) -> bool:
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['close'], row['open'])
    lower_shadow = min(row['close'], row['open']) - row['low']
    return (lower_shadow > 2 * body) and (upper_shadow < body * 0.5) and (row['close'] > row['open'])

def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    return (prev['close'] > prev['open']) and (curr['open'] > prev['close']) and (curr['close'] < prev['open'])

def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    return (prev['close'] < prev['open']) and (curr['open'] < prev['close']) and (curr['close'] > prev['open'])

async def check_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    """Enhanced candlestick pattern detection with historical success rates"""
    if len(df) < 3:
        return []
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    patterns = []
    
    if is_shooting_star(curr):
        pattern_type = "shooting_star"
        success_rate = await get_pattern_success_rate(pattern_type, 30)
        pattern_data = {
            "price": float(curr['close']),
            "open": float(curr['open']),
            "high": float(curr['high']),
            "low": float(curr['low']),
            "volume": float(curr.get('volume', 0))
        }
        await store_pattern(SYMBOL, pattern_type, pattern_data, success_rate)
        patterns.append(f"Shooting Star — Price: {curr['close']:.2f} (Success: {success_rate:.1%})")
        
    if is_hammer(curr):
        pattern_type = "hammer"
        success_rate = await get_pattern_success_rate(pattern_type, 30)
        pattern_data = {
            "price": float(curr['close']),
            "open": float(curr['open']),
            "high": float(curr['high']),
            "low": float(curr['low']),
            "volume": float(curr.get('volume', 0))
        }
        await store_pattern(SYMBOL, pattern_type, pattern_data, success_rate)
        patterns.append(f"Hammer — Price: {curr['close']:.2f} (Success: {success_rate:.1%})")
        
    if is_bearish_engulfing(prev, curr):
        pattern_type = "bearish_engulfing"
        success_rate = await get_pattern_success_rate(pattern_type, 30)
        pattern_data = {
            "prev_candle": {"open": float(prev['open']), "close": float(prev['close'])},
            "curr_candle": {"open": float(curr['open']), "close": float(curr['close'])},
            "price": float(curr['close']),
            "volume": float(curr.get('volume', 0))
        }
        await store_pattern(SYMBOL, pattern_type, pattern_data, success_rate)
        patterns.append(f"Bearish Engulfing — Price: {curr['close']:.2f} (Success: {success_rate:.1%})")
        
    if is_bullish_engulfing(prev, curr):
        pattern_type = "bullish_engulfing"
        success_rate = await get_pattern_success_rate(pattern_type, 30)
        pattern_data = {
            "prev_candle": {"open": float(prev['open']), "close": float(prev['close'])},
            "curr_candle": {"open": float(curr['open']), "close": float(curr['close'])},
            "price": float(curr['close']),
            "volume": float(curr.get('volume', 0))
        }
        await store_pattern(SYMBOL, pattern_type, pattern_data, success_rate)
        patterns.append(f"Bullish Engulfing — Price: {curr['close']:.2f} (Success: {success_rate:.1%})")
        
    return patterns

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return None
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

async def check_rsi_alerts(df: pd.DataFrame) -> str:
    """Enhanced RSI analysis with historical context"""
    rsi = calculate_rsi(df)
    if rsi is None:
        return None
    curr_price = df["close"].iloc[-1]
    
    if rsi > 70:
        # Store overbought pattern
        pattern_data = {
            "rsi": float(rsi),
            "price": float(curr_price),
            "threshold": 70
        }
        success_rate = await get_pattern_success_rate("rsi_overbought", 30)
        await store_pattern(SYMBOL, "rsi_overbought", pattern_data, success_rate)
        return f"RSI Overbought — RSI: {rsi:.2f} Price: {curr_price:.2f} (Historical accuracy: {success_rate:.1%})"
        
    elif rsi < 30:
        # Store oversold pattern
        pattern_data = {
            "rsi": float(rsi),
            "price": float(curr_price),
            "threshold": 30
        }
        success_rate = await get_pattern_success_rate("rsi_oversold", 30)
        await store_pattern(SYMBOL, "rsi_oversold", pattern_data, success_rate)
        return f"RSI Oversold — RSI: {rsi:.2f} Price: {curr_price:.2f} (Historical accuracy: {success_rate:.1%})"
        
    return None

def should_alert(pattern_type: str) -> bool:
    now = time.time()
    last = last_alerts.get(pattern_type)
    if last is None or (now - last) >= ALERT_COOLDOWN:
        last_alerts[pattern_type] = now
        return True
    return False

# ---------- Technical Analysis Summary (Without AI) ----------  
async def get_ai_analysis(df: pd.DataFrame, alerts: List[str], rsi: float) -> str:
    """
    Generate technical analysis summary based on market conditions and historical patterns.
    """
    try:
        if df is None or df.empty:
            return "No data available for analysis."
        
        latest = df.iloc[-1]
        price = float(latest.get("close", 0))
        
        # Calculate additional metrics
        sma_short = df["close"].rolling(window=SHORT_WINDOW).mean().iloc[-1] if len(df) >= SHORT_WINDOW else None
        sma_long = df["close"].rolling(window=LONG_WINDOW).mean().iloc[-1] if len(df) >= LONG_WINDOW else None
        
        # Price trend (last 10 candles)
        recent_prices = df["close"].tail(10).tolist()
        price_change = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100) if len(recent_prices) > 0 else 0
        
        # Get historical context from MongoDB
        trend_analysis = await get_price_trend_analysis(SYMBOL, 24)
        recent_alerts = await get_recent_alerts(SYMBOL, 24)
        historical_patterns = await get_historical_patterns(SYMBOL, days_back=7)
        
        # Calculate pattern success rates
        golden_cross_success = await get_pattern_success_rate("golden_cross", 30)
        death_cross_success = await get_pattern_success_rate("death_cross", 30)
        
        # Format strings safely
        sma_short_str = f"${sma_short:.2f}" if sma_short is not None else "N/A"
        sma_long_str = f"${sma_long:.2f}" if sma_long is not None else "N/A"
        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
        
        # Build enhanced context for AI
        context = f"""Analyze the following cryptocurrency market data for {SYMBOL} with historical context:

CURRENT MARKET STATE:
- Price: ${price:.2f}
- Recent Price Change (10 periods): {price_change:+.2f}%
- RSI (14): {rsi_str} {'(Overbought)' if rsi and rsi > 70 else '(Oversold)' if rsi and rsi < 30 else '(Neutral)'}
- SMA {SHORT_WINDOW}: {sma_short_str}
- SMA {LONG_WINDOW}: {sma_long_str}

HISTORICAL TREND ANALYSIS (24h):
- Trend Direction: {trend_analysis.get('trend', 'unknown')}
- Trend Strength: {trend_analysis.get('strength', 0):.2f}/1.0
- 24h Price Change: ${trend_analysis.get('price_change_24h', 0):.2f}

PATTERN RELIABILITY (30-day success rates):
- Golden Cross patterns: {golden_cross_success:.1%}
- Death Cross patterns: {death_cross_success:.1%}

RECENT ACTIVITY:
- Alerts in last 24h: {len(recent_alerts)}
- Patterns detected (7 days): {len(historical_patterns)}

CURRENT ALERTS:
{chr(10).join('- ' + alert for alert in alerts) if alerts else '- No significant patterns detected'}

Provide comprehensive trading insight (3-4 sentences) covering:
1. Current market sentiment with historical context
2. Pattern reliability and success probability
3. Risk assessment based on recent trends
4. Strategic trading suggestion (not financial advice)"""

        # Store analysis context for future reference
        context_data = {
            "price": price,
            "rsi": rsi,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "trend_analysis": trend_analysis,
            "recent_alerts_count": len(recent_alerts),
            "historical_patterns_count": len(historical_patterns),
            "pattern_success_rates": {
                "golden_cross": golden_cross_success,
                "death_cross": death_cross_success
            }
        }

        # Generate technical analysis summary without AI
        analysis_parts = []
        
        # Market sentiment based on RSI
        if rsi is not None:
            if rsi > 70:
                analysis_parts.append(f"Market shows overbought conditions (RSI: {rsi:.1f}), suggesting potential selling pressure.")
            elif rsi < 30:
                analysis_parts.append(f"Market appears oversold (RSI: {rsi:.1f}), indicating possible buying opportunity.")
            else:
                analysis_parts.append(f"RSI at {rsi:.1f} indicates neutral momentum, waiting for clearer signals.")
        
        # Trend analysis
        trend = trend_analysis.get('trend', 'sideways')
        if trend == 'bullish':
            analysis_parts.append("Short-term trend shows bullish momentum with upward price action.")
        elif trend == 'bearish':
            analysis_parts.append("Recent price action indicates bearish pressure and downward trend.")
        else:
            analysis_parts.append("Price is moving sideways with no clear directional bias.")
        
        # Moving average analysis
        if sma_short is not None and sma_long is not None:
            if sma_short > sma_long:
                analysis_parts.append(f"Short-term MA (${sma_short:.2f}) above long-term MA (${sma_long:.2f}) suggests bullish structure.")
            else:
                analysis_parts.append(f"Short-term MA (${sma_short:.2f}) below long-term MA (${sma_long:.2f}) indicates bearish structure.")
        
        # Recent activity context
        if len(recent_alerts) > 0:
            analysis_parts.append(f"Recent activity: {len(recent_alerts)} alerts in 24h, {len(historical_patterns)} patterns in 7 days.")
        
        # Current alerts summary
        if alerts:
            alert_summary = f"Active signals: {', '.join(alerts[:2])}"  # Show first 2 alerts
            if len(alerts) > 2:
                alert_summary += f" and {len(alerts) - 2} more."
            analysis_parts.append(alert_summary)
        
        analysis_result = " ".join(analysis_parts)
        
        # Store the technical analysis in MongoDB
        context_data["analysis_type"] = "technical_summary"
        await store_ai_analysis(SYMBOL, analysis_result, context_data)
        
        return analysis_result
    
    except Exception as e:
        error_msg = f"AI analysis unavailable: {str(e)}"
        print(f"AI Analysis Error: {e}")
        return error_msg

# ---------- Agent / monitor loop (async) ----------
async def agent_loop():
    iteration = 0
    while True:
        iteration += 1
        try:
            # fetch data in thread to avoid blocking
            df = await asyncio.to_thread(get_crypto_data_sync, SYMBOL, 500)
            if df is None or df.empty:
                # broadcast heartbeat with no data
                heartbeat_payload = {
                    "type": "heartbeat",
                        "timestamp": datetime.now(pytz.UTC).isoformat(),
                    "message": "no data"
                }
                print(f"\nWebSocket Heartbeat - Iteration {iteration}")
                print(f"Status: No market data available")
                print(f"Connected Clients: {len(manager.active_connections)}")
                print(f"Timestamp: {heartbeat_payload['timestamp']}")
                print("-" * 40)
                await manager.broadcast(heartbeat_payload)
            else:
                alerts = []
                
                # Check patterns with historical analysis
                cross_alert = await check_cross(df)
                if cross_alert and should_alert("cross"):
                    alerts.append(cross_alert)
                    await store_alert(SYMBOL, "cross", cross_alert)

                candle_alerts = await check_candlestick_patterns(df)
                if candle_alerts:
                    for _ in candle_alerts:
                        if should_alert("candlestick"):
                            alerts.extend(candle_alerts)
                            for alert in candle_alerts:
                                await store_alert(SYMBOL, "candlestick", alert)
                            break

                rsi_alert = await check_rsi_alerts(df)
                if rsi_alert and should_alert("rsi"):
                    alerts.append(rsi_alert)
                    await store_alert(SYMBOL, "rsi", rsi_alert)

                # Calculate indicators
                rsi_value = calculate_rsi(df)
                sma_short_value = df["close"].rolling(window=SHORT_WINDOW).mean().iloc[-1] if len(df) >= SHORT_WINDOW else None
                sma_long_value = df["close"].rolling(window=LONG_WINDOW).mean().iloc[-1] if len(df) >= LONG_WINDOW else None
                
                latest = df.iloc[-1].to_dict()
                
                # Get volume data
                volume_data = latest.get("volume", 0)
                
                # Store market data in MongoDB
                
                bar_data = {
                    "t": latest.get("timestamp") or latest.get("time"),
                    "o": float(latest.get("open", 0)),
                    "h": float(latest.get("high", 0)),
                    "l": float(latest.get("low", 0)),
                    "c": float(latest.get("close", 0)),
                    "v": float(volume_data) if volume_data is not None else 0,
                }
                
                indicators = {
                    "rsi": rsi_value,
                    "sma_short": sma_short_value,
                    "sma_long": sma_long_value,
                    "price_change": 0  # Will be calculated with historical data
                }
                
                await store_market_data(SYMBOL, bar_data, indicators)

                # Get enhanced AI analysis with historical context
                ai_analysis = await get_ai_analysis(df, alerts, rsi_value)

                payload = {
                    "type": "update",
                        "timestamp": datetime.now(pytz.UTC).isoformat(),
                    "symbol": SYMBOL,
                    "price": float(latest.get("close", 0)),
                    "bar": bar_data,
                    "rsi": rsi_value,
                    "sma_short": sma_short_value,
                    "sma_long": sma_long_value,
                    "alerts": alerts,
                    "ai_analysis": ai_analysis,
                    "iteration": iteration,
                    "historical_context": {
                        "alerts_24h": len(await get_recent_alerts(SYMBOL, 24)),
                        "patterns_7d": len(await get_historical_patterns(SYMBOL, days_back=7))
                    }
                }
                
                # Print WebSocket payload for debugging
                rsi_display = f"{payload['rsi']:.2f}" if payload['rsi'] is not None else "N/A"
                sma_short_display = f"${payload['sma_short']:.2f}" if payload['sma_short'] is not None else "N/A"
                sma_long_display = f"${payload['sma_long']:.2f}" if payload['sma_long'] is not None else "N/A"
                alerts_display = f"{len(alerts)} ({', '.join(alerts[:2])})" if alerts else "0 (None)"
                
                print(f"\nWebSocket Broadcast - Iteration {iteration}")
                print(f"Symbol: {payload['symbol']}")
                print(f"Price: ${payload['price']:.2f}")
                print(f"RSI: {rsi_display}")
                print(f"SMA Short: {sma_short_display}")
                print(f"SMA Long: {sma_long_display}")
                print(f"Volume: {bar_data['v']}")
                print(f"Alerts: {alerts_display}")
                print(f"Analysis: {ai_analysis[:80]}{'...' if len(ai_analysis) > 80 else ''}")
                print(f"Connected Clients: {len(manager.active_connections)}")
                print(f"Timestamp: {payload['timestamp']}")
                print("-" * 60)
                
                await manager.broadcast(payload)
        except Exception as e:
            error_payload = {
                "type": "error",
                    "timestamp": datetime.now(pytz.UTC).isoformat(),
                "message": str(e)
            }
            print(f"\nWebSocket Error - Iteration {iteration}")
            print(f"Error: {str(e)}")
            print(f"Connected Clients: {len(manager.active_connections)}")
            print(f"Timestamp: {error_payload['timestamp']}")
            print("-" * 40)
            await manager.broadcast(error_payload)
        await asyncio.sleep(CHECK_INTERVAL)

# ---------- FastAPI events and routes ----------
@app.get("/status")
async def status():
    return {"status": "running", "symbol": SYMBOL, "short_window": SHORT_WINDOW, "long_window": LONG_WINDOW}

@app.get("/historical-analysis")
async def get_historical_analysis():
    """Get comprehensive historical analysis"""
    try:
        # Get recent data
        market_data = await get_historical_market_data(SYMBOL, 24, 100)
        alerts = await get_recent_alerts(SYMBOL, 24)
        patterns = await get_historical_patterns(SYMBOL, days_back=7)
        trend_analysis = await get_price_trend_analysis(SYMBOL, 24)
        
        # Calculate pattern success rates
        pattern_success_rates = {
            "golden_cross": await get_pattern_success_rate("golden_cross", 30),
            "death_cross": await get_pattern_success_rate("death_cross", 30),
            "shooting_star": await get_pattern_success_rate("shooting_star", 30),
            "hammer": await get_pattern_success_rate("hammer", 30),
            "rsi_overbought": await get_pattern_success_rate("rsi_overbought", 30),
            "rsi_oversold": await get_pattern_success_rate("rsi_oversold", 30)
        }
        
        return {
            "symbol": SYMBOL,
            "timestamp": datetime.utcnow().isoformat(),
            "market_data_points": len(market_data),
            "alerts_24h": len(alerts),
            "patterns_7d": len(patterns),
            "trend_analysis": trend_analysis,
            "pattern_success_rates": pattern_success_rates,
            "recent_alerts": [{"timestamp": alert["timestamp"].isoformat(), 
                             "type": alert["alert_type"], 
                             "message": alert["message"]} for alert in alerts[:5]],
            "recent_patterns": [{"timestamp": pattern["timestamp"].isoformat(),
                               "type": pattern["pattern_type"],
                               "confidence": pattern["confidence"]} for pattern in patterns[:5]]
        }
    except Exception as e:
        return {"error": f"Failed to get historical analysis: {str(e)}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # simple listener — we keep connection open and client may send messages (not required)
        while True:
            data = await websocket.receive_text()
            # Echo or handle simple commands from client (e.g., "ping")
            if data.lower() == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Run the FastAPI app object directly instead of using an import string
    # (previously attempted to import module "app" which doesn't exist in this repo)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
