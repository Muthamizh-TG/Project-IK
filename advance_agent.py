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

import numpy as np

# Simple FastAPI app (no MongoDB lifespan)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start agent loop on FastAPI startup
@app.on_event("startup")
async def start_agent_loop():
    asyncio.create_task(agent_loop())

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


if not API_KEY or not API_SECRET:
    raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env")

# Alpaca client
client = CryptoHistoricalDataClient(API_KEY, API_SECRET)





# Simple FastAPI app (no MongoDB lifespan)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start agent loop on FastAPI startup
@app.on_event("startup")
async def start_agent_loop():
    asyncio.create_task(agent_loop())

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



async def get_price_trend_analysis(symbol: str, hours_back: int = 24):
    """Analyze price trends from historical data"""
    # MongoDB is removed, so we cannot fetch historical data
    # Return stub response
    return {"trend": "insufficient_data", "strength": 0.0}

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
        return f"Golden Cross — Potential uptrend. Price: {curr_price:.2f}"
    elif prev_short > prev_long and curr_short < curr_long:
        # Death Cross detected
        return f"Death Cross — Potential downtrend. Price: {curr_price:.2f}"
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
        patterns.append(f"Shooting Star — Price: {curr['close']:.2f}")
    if is_hammer(curr):
        patterns.append(f"Hammer — Price: {curr['close']:.2f}")
    if is_bearish_engulfing(prev, curr):
        patterns.append(f"Bearish Engulfing — Price: {curr['close']:.2f}")
    if is_bullish_engulfing(prev, curr):
        patterns.append(f"Bullish Engulfing — Price: {curr['close']:.2f}")
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
        return f"RSI Overbought — RSI: {rsi:.2f} Price: {curr_price:.2f}"
    elif rsi < 30:
        return f"RSI Oversold — RSI: {rsi:.2f} Price: {curr_price:.2f}"
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
        
        # Generate technical analysis summary without MongoDB context
        sma_short_str = f"${sma_short:.2f}" if sma_short is not None else "N/A"
        sma_long_str = f"${sma_long:.2f}" if sma_long is not None else "N/A"
        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"

        analysis_parts = []
        if rsi is not None:
            if rsi > 70:
                analysis_parts.append(f"Market shows overbought conditions (RSI: {rsi:.1f}), suggesting potential selling pressure.")
            elif rsi < 30:
                analysis_parts.append(f"Market appears oversold (RSI: {rsi:.1f}), indicating possible buying opportunity.")
            else:
                analysis_parts.append(f"RSI at {rsi:.1f} indicates neutral momentum, waiting for clearer signals.")
        if sma_short is not None and sma_long is not None:
            if sma_short > sma_long:
                analysis_parts.append(f"Short-term MA (${sma_short:.2f}) above long-term MA (${sma_long:.2f}) suggests bullish structure.")
            else:
                analysis_parts.append(f"Short-term MA (${sma_short:.2f}) below long-term MA (${sma_long:.2f}) indicates bearish structure.")
        if alerts:
            alert_summary = f"Active signals: {', '.join(alerts[:2])}"
            if len(alerts) > 2:
                alert_summary += f" and {len(alerts) - 2} more."
            analysis_parts.append(alert_summary)
        analysis_result = " ".join(analysis_parts)
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
                ist = pytz.timezone('Asia/Kolkata')
                ist_now = datetime.now(ist)
                heartbeat_payload = {
                    "type": "heartbeat",
                    "timestamp": ist_now.isoformat(),
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
                # Check patterns
                cross_alert = await check_cross(df)
                if cross_alert and should_alert("cross"):
                    alerts.append(cross_alert)
                candle_alerts = await check_candlestick_patterns(df)
                if candle_alerts:
                    for _ in candle_alerts:
                        if should_alert("candlestick"):
                            alerts.extend(candle_alerts)
                            break
                rsi_alert = await check_rsi_alerts(df)
                if rsi_alert and should_alert("rsi"):
                    alerts.append(rsi_alert)
                # Calculate indicators
                rsi_value = calculate_rsi(df)
                sma_short_value = df["close"].rolling(window=SHORT_WINDOW).mean().iloc[-1] if len(df) >= SHORT_WINDOW else None
                sma_long_value = df["close"].rolling(window=LONG_WINDOW).mean().iloc[-1] if len(df) >= LONG_WINDOW else None
                latest = df.iloc[-1].to_dict()
                volume_data = latest.get("volume", 0)
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
                    "price_change": 0
                }
                ai_analysis = await get_ai_analysis(df, alerts, rsi_value)
                ist = pytz.timezone('Asia/Kolkata')
                ist_now = datetime.now(ist)
                payload = {
                    "type": "update",
                    "timestamp": ist_now.isoformat(),
                    "symbol": SYMBOL,
                    "price": float(latest.get("close", 0)),
                    "bar": bar_data,
                    "rsi": rsi_value,
                    "sma_short": sma_short_value,
                    "sma_long": sma_long_value,
                    "alerts": alerts,
                    "ai_analysis": ai_analysis,
                    "iteration": iteration
                }
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
            ist = pytz.timezone('Asia/Kolkata')
            ist_now = datetime.now(ist)
            error_payload = {
                "type": "error",
                "timestamp": ist_now.isoformat(),
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


# Remove /historical-analysis endpoint (MongoDB dependent)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # simple listener — we keep connection open and client may send messages (not required)
        while True:
            data = await websocket.receive_text()
            # Echo or handle simple commands from client (e.g., "ping")
            if data.lower() == "ping":
                ist = pytz.timezone('Asia/Kolkata')
                ist_now = datetime.now(ist)
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": ist_now.isoformat()}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Run the FastAPI app object directly instead of using an import string
    # (previously attempted to import module "app" which doesn't exist in this repo)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
