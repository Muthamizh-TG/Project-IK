# ============================================================================
# MULTI-CRYPTO LIVE CANDLE FEED FOR YOUR REACT APP (1:1 MATCH)
# Sends { type: "live_bars", timestamp: "...", bars: [...] }
# ============================================================================
import json
import asyncio
import logging
import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import alpaca_trade_api as alpaca
from alpaca_trade_api.rest import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("feed")

# ============================================================================
# ALPACA CONFIG
# ============================================================================
API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
API_SECRET = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA = alpaca.REST(API_KEY, API_SECRET, BASE_URL)

# Default symbols for your dashboard
DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD"]

# Connected clients state
CLIENTS = []


# ============================================================================
# FASTAPI SETUP
# ============================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def now_utc_min():
    return pd.Timestamp.utcnow().floor("1min")


def to_crypto_api_symbol(sym):
    """Convert BTCUSD → BTC/USD for Alpaca crypto endpoint."""
    return sym[:-3] + "/USD" if sym.endswith("USD") else sym


# ============================================================================
# FETCH 1-MIN BARS FOR MULTIPLE SYMBOLS
# ============================================================================
async def fetch_bars(symbols, last_ts_map):
    utc_now = now_utc_min()
    all_bars = []

    for sym in symbols:
        try:
            api_symbol = to_crypto_api_symbol(sym)

            last_ts = last_ts_map.get(sym, None)

            # First run → initialize timestamp
            if last_ts is None:
                last_ts_map[sym] = utc_now
                continue

            # No new minute yet
            if utc_now <= last_ts:
                continue

            start = (last_ts + pd.Timedelta("1min")).strftime("%Y-%m-%dT%H:%M:%SZ")
            end = utc_now.strftime("%Y-%m-%dT%H:%M:%SZ")

            data = ALPACA.get_crypto_bars(
                api_symbol, TimeFrame.Minute, start=start, end=end
            )
            df = getattr(data, "df", data)

            if df is None or len(df) == 0:
                continue

            for ts, row in df.iterrows():
                last_ts_map[sym] = ts

                # Convert time to IST
                local_ts = ts.tz_convert("Asia/Kolkata")

                bar = {
                    "symbol": api_symbol,  # Frontend expects "BTC/USD"
                    "timestamp": str(local_ts),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                all_bars.append(bar)

        except Exception as e:
            log.error(f"Fetch error for {sym}: {e}")

    return all_bars, utc_now


# ============================================================================
# BROADCAST TO ALL CLIENTS
# ============================================================================
async def broadcast_to_all(message: dict):
    dead = []
    for ws in CLIENTS:
        try:
            await ws.send_text(json.dumps(message))
        except:
            dead.append(ws)
    for d in dead:
        CLIENTS.remove(d)


# ============================================================================
# BACKGROUND LIVE LOOP
# ============================================================================
async def live_loop():
    last_ts_map = {sym: None for sym in DEFAULT_SYMBOLS}

    while True:
        try:
            bars, utc_now = await fetch_bars(DEFAULT_SYMBOLS, last_ts_map)

            if bars:
                message = {
                    "type": "live_bars",
                    "timestamp": str(utc_now.tz_convert("Asia/Kolkata")),
                    "bars": bars,
                }

                log.info(f"Sending {len(bars)} bars")
                await broadcast_to_all(message)
            else:
                # keep connection alive
                await broadcast_to_all({"type": "heartbeat"})

        except Exception as e:
            log.error(f"Background loop error: {e}")

        await asyncio.sleep(5)  # Polling interval


# Start background loop on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(live_loop())


# ============================================================================
# WEBSOCKET ENDPOINT USED BY YOUR REACT APP
# ============================================================================
@app.websocket("/ws/simple")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.append(ws)
    log.info("Client connected to /ws/simple")

    try:
        while True:
            msg = await ws.receive_text()
            # React sends {"symbols": [...]}, but we ignore (multi-feed enabled)
    except:
        pass
    finally:
        CLIENTS.remove(ws)
        log.info("Client disconnected")

# ============================================================================
# RUN UVICORN IF EXECUTED DIRECTLY
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test2:app", host="0.0.0.0", port=8000, reload=True)
