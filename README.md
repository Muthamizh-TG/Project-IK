# Multi-Crypto Live Chart & Agent

## Overview

This project provides a real-time multi-cryptocurrency dashboard with live price charts, RSI-based trading signals, and manual trading tools. It consists of:

- **FastAPI Backend**: Streams live 1-minute crypto bars via WebSocket.
- **React Frontend**: Displays interactive charts, stats, and trading signals.

## Features

- Live 1-minute OHLCV bars for BTC, ETH, SOL, XRP, DOGE.
- RSI calculation and buy/sell signal detection.
- Toggle cryptos on/off, view price or % change.
- Manual trading scripts (buy/sell/cancel/check status).
- Responsive UI with real-time updates.

## Structure

```
Stock_Agent/
  advance_agent.py
  main.py
  test2.py           # FastAPI live feed server
  websocket_server.py
  manual mode/
    manual_buy.py
    manual_sell.py
    ...
  mover/
    mover_v1.py
    ...
  alpaca-agent-ui/
    src/
      MultiCryptoChart.jsx
      ...
    public/
    package.json
    ...
```

## Getting Started

### Backend (FastAPI)

1. Install dependencies:
   ```
   pip install fastapi uvicorn pandas alpaca-trade-api
   ```
2. Run the server:
   ```
   python websocket_server.py
   ```
   - WebSocket endpoint: `ws://localhost:8082/ws/simple`

### Frontend (React)

1. Go to `alpaca-agent-ui` folder.
2. Install dependencies:
   ```
   npm install
   ```
3. Start the app:
   ```
   npm run dev
   ```
   - Access at: `http://localhost:5173` (or as shown in terminal)

## Manual Trading

Scripts in `manual mode/` allow manual buy/sell/cancel operations using Alpaca API. Edit API keys as needed.

## Configuration

- Update API keys in `test2.py` and manual scripts.
- Default symbols: BTC/USD, ETH/USD, SOL/USD, XRP/USD, DOGE/USD.

## Notes

- Chart resets on each app load (no stale data).
- Handles missing bars by filling with last known value.

## License

MIT
