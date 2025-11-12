# Data Update Frequency Configuration

## Summary of Changes

### Backend (websocket_server.py)

#### 1. **New Historical Data Fetcher Method**
```python
async def fetch_historical_crypto_data(self, minutes=15)
```
- Fetches past 15 minutes of 1-minute bars for each cryptocurrency
- Returns complete bar data (open, high, low, close, volume) for each minute
- Called once when a client first connects

#### 2. **Updated WebSocket Endpoint**
- When a client connects, immediately sends **15 minutes of historical data**
- Message type: `"historical_data"`
- Includes all OHLCV data for each bar in the historical period

#### 3. **Broadcast Interval Changed**
- **OLD**: Every 60 seconds (1 minute)
- **NEW**: Every 300 seconds (5 minutes)
- Reduces data load while still providing frequent updates

---

## Data Flow

### Initial Connection (Client Joins)
```
Client Connects → Server sends "historical_data" (last 15 minutes)
                → Chart immediately populates with 15 data points
```

### Continuous Updates (Every 5 minutes)
```
Every 300 seconds → Server sends "multi_crypto_update" (latest bar)
                  → Chart appends new point
                  → Keeps last 100 points (500 minutes ≈ 8+ hours)
```

---

## Message Types

### 1. Historical Data (On Connection)
```json
{
  "type": "historical_data",
  "timestamp": "2025-11-12T10:30:45.123456",
  "data": {
    "BTC/USD": {
      "bars": [
        {"time": "10:15:32", "price": 42150.50, "open": 42100, "high": 42200, "low": 42100, "volume": 15000},
        {"time": "10:20:32", "price": 42175.25, "open": 42150, "high": 42180, "low": 42145, "volume": 12000},
        ...
      ],
      "price": 42200.00,
      "change": 0.12,
      "available": true
    }
  }
}
```

### 2. Update Data (Every 5 Minutes)
```json
{
  "type": "multi_crypto_update",
  "timestamp": "2025-11-12T10:35:00.000000",
  "data": {
    "BTC/USD": {
      "symbol": "BTC/USD",
      "price": 42220.00,
      "open": 42200.00,
      "high": 42250.00,
      "low": 42150.00,
      "volume": 18500,
      "change": 0.17
    }
  }
}
```

---

## Frontend (MultiCryptoChart.jsx)

### Updated Message Handlers

1. **On "historical_data" message**:
   - Loads all 15 historical bars
   - Sets baseline prices from first bar
   - Calculates percentage changes
   - Populates chart immediately
   - Saves to localStorage

2. **On "multi_crypto_update" message**:
   - Appends new 5-minute bar to chart
   - Updates baseline prices if needed
   - Recalculates percentage changes
   - Maintains last 100 points

---

## Benefits

✅ **Faster Initial Load**: Chart shows 15 minutes of data immediately  
✅ **Reduced Server Load**: Updates every 5 minutes instead of every 1 minute  
✅ **Better Data Persistence**: localStorage keeps historical data across sessions  
✅ **Complete OHLCV Data**: Each bar includes open, high, low, close, volume  
✅ **Flexible Updates**: Easy to change interval by modifying `await asyncio.sleep(300)`  

---

## Testing

### Start Backend
```bash
python websocket_server.py
```

### Start Frontend
```bash
cd alpaca-agent-ui
npm run dev
```

### Expected Behavior
1. Open http://localhost:5173
2. Navigate to "Multi-Crypto Chart" (or XRP Chart)
3. **Immediate Result**: Chart loads with 15 minutes of past data
4. **After 5 minutes**: New data point added to chart
5. **Every 5 minutes**: New updates continue

---

## Configuration

To change update frequency, modify in `websocket_server.py`:

### Line ~300 (broadcast_crypto_data function)
```python
# Wait 300 seconds (5 minutes) before next update
await asyncio.sleep(300)  # Change this number for different intervals
```

### Line ~220 (fetch_historical_crypto_data call)
```python
# On connection, send past 15 minutes of historical data
historical_data = await fetcher.fetch_historical_crypto_data(minutes=15)  # Change 15 to desired minutes
```

---

## File Changes

- ✅ `websocket_server.py` - Added historical data fetcher and updated broadcast interval
- ✅ `MultiCryptoChart.jsx` - Added historical data handler and improved message processing
