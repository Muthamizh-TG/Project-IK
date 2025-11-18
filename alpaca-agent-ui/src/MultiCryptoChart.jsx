// src/MultiCryptoChart.jsx
import React, { useEffect, useState, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
  ComposedChart,
  ReferenceLine,
  ScatterChart,
  Scatter,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Wifi,
  WifiOff,
  RefreshCw,
  XCircle,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Activity,
  DollarSign,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

// ============================================================================
// RSI CALCULATION FUNCTION
// ============================================================================
const calculateRSI = (prices, period = 14) => {
  if (prices.length < period + 1) return null;

  let gains = 0;
  let losses = 0;

  // Calculate average gains and losses
  for (let i = prices.length - period; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) {
      gains += change;
    } else {
      losses += Math.abs(change);
    }
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;

  if (avgLoss === 0) {
    return avgGain === 0 ? 50 : 100;
  }

  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  return rsi;
};

// ============================================================================
// BUY/SELL SIGNAL DETECTOR (Based on RSI)
// ============================================================================
const detectSignals = (chartData, symbol, period = 14) => {
  if (!chartData || chartData.length < period + 1) return [];

  const prices = chartData.map(d => d[symbol]).filter(p => p !== undefined);
  const signals = [];

  // Calculate RSI for the last point
  const rsi = calculateRSI(prices, period);

  if (rsi !== null) {
    // BUY Signal: RSI < 30 (Oversold)
    if (rsi < 30) {
      signals.push({
        type: 'BUY',
        rsi: rsi,
        index: chartData.length - 1
      });
    }
    // SELL Signal: RSI > 70 (Overbought)
    else if (rsi > 70) {
      signals.push({
        type: 'SELL',
        rsi: rsi,
        index: chartData.length - 1
      });
    }
  }

  return signals;
};

export default function MultiCryptoChart() {
  const [chartData, setChartData] = useState(() => {
    // Load persisted data from localStorage on initialization
    const saved = localStorage.getItem('multiCryptoChartData');
    return saved ? JSON.parse(saved) : [];
  });
  const [cryptoStats, setCryptoStats] = useState({});
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [baselinePrices, setBaselinePrices] = useState(() => {
    // Load persisted baselines from localStorage
    const saved = localStorage.getItem('multiCryptoBaselines');
    return saved ? JSON.parse(saved) : {};
  }); // Store first price for each crypto
  const [viewMode, setViewMode] = useState("percentage"); // "percentage" or "absolute"
  const [selectedCryptos, setSelectedCryptos] = useState({
    'BTC/USD': true,
    'ETH/USD': true,
    'SOL/USD': true,
    'XRP/USD': true,
    'DOGE/USD': true,
  });
  const [tradingSignals, setTradingSignals] = useState({}); // Store signals for each crypto
  const wsRef = useRef(null);

  // Crypto configuration with colors - XRP focused
  const CRYPTO_CONFIG = {
    'BTC/USD': { color: '#F7931A', name: 'Bitcoin', symbol: 'BTC' },
    'ETH/USD': { color: '#627EEA', name: 'Ethereum', symbol: 'ETH' },
    'SOL/USD': { color: '#14F195', name: 'Solana', symbol: 'SOL' },
    'XRP/USD': { color: '#2196f3', name: 'Ripple', symbol: 'XRP' },
    'DOGE/USD': { color: '#C2A633', name: 'Dogecoin', symbol: 'DOGE' },
    // 'HYPER/USD': { color: '#FF6B6B', name: 'Hyper', symbol: 'HYPER' },
    // 'PEPENODE/USD': { color: '#4ECDC4', name: 'PepeNode', symbol: 'PEPENODE' },
    // 'BEST/USD': { color: '#95E1D3', name: 'Best', symbol: 'BEST' },
    // 'LILPEPE/USD': { color: '#F38181', name: 'LilPepe', symbol: 'LILPEPE' },
    // 'BONK/USD': { color: '#AA96DA', name: 'Bonk', symbol: 'BONK' },
  };

  useEffect(() => {
      const wsUrl = `ws://localhost:8000/ws/simple`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected");
        setConnectionStatus("connected");
        // Send selected symbols to backend
        const activeSymbols = Object.keys(CRYPTO_CONFIG).filter(s => selectedCryptos[s]);
        ws.send(JSON.stringify({ symbols: activeSymbols }));
      };

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "live_bars" && Array.isArray(msg.bars)) {
            // Process live bars from /ws/simple
            console.log("📊 Live bars update received:", msg.timestamp);
            const timestamp = new Date(msg.timestamp);
            const hours = timestamp.getHours();
            const minutes = timestamp.getMinutes();
            const seconds = timestamp.getSeconds();
            const period = hours >= 12 ? 'PM' : 'AM';
            const displayHours = hours % 12 || 12;
            const timeLabel = `${displayHours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')} ${period}`;

            // Aggregate bars by symbol
            const newDataPoint = {
              time: timeLabel,
              displayTime: timeLabel,
              timestamp: msg.timestamp,
            };
            msg.bars.forEach(bar => {
              const symbol = bar.symbol;
              newDataPoint[symbol] = bar.close;
              newDataPoint[`${symbol}_close`] = bar.close;
              newDataPoint[`${symbol}_open`] = bar.open;
              newDataPoint[`${symbol}_high`] = bar.high;
              newDataPoint[`${symbol}_low`] = bar.low;
              newDataPoint[`${symbol}_volume`] = bar.volume;
            });

            // Update baseline prices for initial setup
            setBaselinePrices((prevBaseline) => {
              const newBaseline = { ...prevBaseline };
              msg.bars.forEach(bar => {
                const symbol = bar.symbol;
                if (!newBaseline[symbol]) {
                  newBaseline[symbol] = bar.close;
                  console.log(`✅ Baseline set for ${symbol}: ${newBaseline[symbol]}`);
                }
                if (newBaseline[symbol]) {
                  newDataPoint[`${symbol}_pct`] = ((bar.close - newBaseline[symbol]) / newBaseline[symbol]) * 100;
                } else {
                  newDataPoint[`${symbol}_pct`] = 0;
                }
              });
              localStorage.setItem('multiCryptoBaselines', JSON.stringify(newBaseline));
              return newBaseline;
            });

            // Append to chart data (keep last 120 points)
            setChartData((prevData) => {
              const next = [...prevData, newDataPoint].slice(-120);
              const indexedData = next.map((point, idx) => ({ ...point, index: idx }));
              localStorage.setItem('multiCryptoChartData', JSON.stringify(indexedData));
              console.log(`📈 Chart data points: ${indexedData.length}`);
              return indexedData;
            });

            // Update crypto statistics
            const stats = {};
            msg.bars.forEach(bar => {
              stats[bar.symbol] = {
                price: bar.close,
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
                volume: bar.volume,
                available: true,
                change: newDataPoint[`${bar.symbol}_pct`] || 0
              };
            });
            setCryptoStats(stats);
            setConnectionStatus("connected");
          } else if (msg.type === "heartbeat") {
            setConnectionStatus("connected");
          } else if (msg.type === "error") {
            console.error("Server error:", msg.message);
            setConnectionStatus("error");
          }
        } catch (e) {
          console.error("Invalid message", e);
          setConnectionStatus("error");
        }
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
        setConnectionStatus("disconnected");
      };

      ws.onerror = () => {
        setConnectionStatus("error");
      };

      return () => {
        ws.close();
      };
    }, [selectedCryptos]);

  // Calculate trading signals whenever chartData updates (and has enough points)
  useEffect(() => {
    if (chartData && chartData.length >= 15) {
      const newSignals = {};
      
      // Calculate RSI for each symbol and add to chartData
      const dataWithRSI = chartData.map((point, idx) => {
        const newPoint = { ...point };
        
        Object.keys(CRYPTO_CONFIG).forEach((symbol) => {
          const prices = chartData.slice(0, idx + 1).map(p => p[symbol]).filter(p => p !== undefined);
          if (prices.length >= 15) {
            const rsi = calculateRSI(prices, 14);
            newPoint[`${symbol}_rsi`] = rsi;
          }
        });
        
        return newPoint;
      });
      
      // Calculate signals from the data
      Object.keys(CRYPTO_CONFIG).forEach((symbol) => {
        const signals = detectSignals(chartData, symbol);
        if (signals.length > 0) {
          newSignals[symbol] = signals[signals.length - 1];
        }
      });
      
      setTradingSignals(newSignals);
      
      // Update chartData with RSI values
      setChartData(dataWithRSI);
    }
  }, [chartData.length]);

  const toggleCrypto = (symbol) => {
    setSelectedCryptos(prev => ({
      ...prev,
      [symbol]: !prev[symbol]
    }));
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case "connected": return <Wifi size={16} color="#4caf50" />;
      case "connecting": return <RefreshCw size={16} className="animate-spin" />;
      case "disconnected": return <WifiOff size={16} color="#f44336" />;
      case "error": return <XCircle size={16} color="#f44336" />;
      default: return <RefreshCw size={16} />;
    }
  };

  const formatPrice = (price) => {
    if (price >= 1000) {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }).format(price);
    }
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price);
  };

  const calculateRSIForChart = (symbol) => {
    if (!chartData || chartData.length < 15) return [];
    
    const prices = chartData.map(d => d[symbol]).filter(p => p !== undefined);
    const rsiData = [];
    
    for (let i = 0; i < chartData.length; i++) {
      const slicedPrices = prices.slice(0, i + 1);
      if (slicedPrices.length >= 15) {
        const rsi = calculateRSI(slicedPrices, 14);
        rsiData.push({
          ...chartData[i],
          rsi: rsi
        });
      } else {
        rsiData.push({
          ...chartData[i],
          rsi: null
        });
      }
    }
    
    return rsiData;
  };

  const fillTimeGap = (lastPoint, newPoint) => {
    /**
     * Fills the gap between last historical point and new update point
     * Creates intermediate data points with the last known prices
     */
    if (!lastPoint || !newPoint) return [];
    
    const lastTime = new Date(lastPoint.timestamp);
    const newTime = new Date(newPoint.timestamp);
    const timeDiffMs = newTime.getTime() - lastTime.getTime();
    const timeDiffMinutes = Math.floor(timeDiffMs / (1000 * 60));
    
    // Only fill gap if it's more than 1 minute but less than 60 minutes
    if (timeDiffMinutes <= 1 || timeDiffMinutes > 60) {
      return [];
    }
    
    const fillerPoints = [];
    
    // Create intermediate points every minute
    for (let i = 1; i < timeDiffMinutes; i++) {
      const fillerTime = new Date(lastTime.getTime() + i * 60 * 1000);
      const hours = fillerTime.getHours();
      const minutes = fillerTime.getMinutes();
      const period = hours >= 12 ? 'PM' : 'AM';
      const displayHours = hours % 12 || 12;
      const timeLabel = `${displayHours}:${String(minutes).padStart(2, '0')} ${period}`;
      
      // Copy last point data but with new time
      const fillerPoint = {
        ...lastPoint,
        time: timeLabel,
        displayTime: timeLabel,
        timestamp: fillerTime.toISOString()
      };
      
      fillerPoints.push(fillerPoint);
    }
    
    return fillerPoints;
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      // Get the data point for this displayTime
      const dataPoint = chartData.find(d => d.displayTime === label);
      // Filter unique symbols
      const seen = new Set();
      const filteredPayload = payload.filter(entry => {
        const symbol = entry.dataKey.replace('_pct', '');
        if (seen.has(symbol)) return false;
        seen.add(symbol);
        return true;
      });
      return (
        <div style={{
          background: 'rgba(0, 0, 0, 0.95)',
          padding: '15px',
          borderRadius: '10px',
          color: 'white',
          fontSize: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
        }}>
          <p style={{ marginBottom: '10px', fontWeight: 'bold', borderBottom: '1px solid #333', paddingBottom: '5px' }}>
            {label}
          </p>
          {filteredPayload.map((entry, index) => {
            const symbol = entry.dataKey.replace('_pct', '');
            const config = CRYPTO_CONFIG[symbol];
            if (!config) return null;
            const price = dataPoint ? dataPoint[symbol] : null;
            const open = dataPoint ? dataPoint[`${symbol}_open`] : null;
            const high = dataPoint ? dataPoint[`${symbol}_high`] : null;
            const low = dataPoint ? dataPoint[`${symbol}_low`] : null;
            const volume = dataPoint ? dataPoint[`${symbol}_volume`] : null;
            return (
              <div key={index} style={{ 
                marginBottom: '8px',
                paddingBottom: '8px',
                borderBottom: index < filteredPayload.length - 1 ? '1px solid #222' : 'none'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  marginBottom: '5px',
                  fontWeight: 'bold'
                }}>
                  <span style={{ 
                    width: '12px', 
                    height: '12px', 
                    backgroundColor: entry.color,
                    borderRadius: '50%',
                    marginRight: '8px'
                  }}></span>
                  <span style={{ color: entry.color }}>{config.name}</span>
                </div>
                {viewMode === "percentage" ? (
                  <div style={{ paddingLeft: '20px', fontSize: '11px' }}>
                    <div>Change: <span style={{ fontWeight: 'bold', color: entry.value >= 0 ? '#4caf50' : '#f44336' }}>
                      {entry.value >= 0 ? '+' : ''}{entry.value.toFixed(3)}%
                    </span></div>
                    <div>Price: <span style={{ fontWeight: 'bold' }}>{price ? formatPrice(price) : 'N/A'}</span></div>
                  </div>
                ) : (
                  <div style={{ paddingLeft: '20px', fontSize: '11px' }}>
                    <div>Price: <span style={{ fontWeight: 'bold' }}>{price ? formatPrice(price) : 'N/A'}</span></div>
                    {open && <div>Open: {formatPrice(open)}</div>}
                    {high && <div>High: {formatPrice(high)}</div>}
                    {low && <div>Low: {formatPrice(low)}</div>}
                    {volume && <div>Volume: {volume >= 1000000 
                      ? `${(volume / 1000000).toFixed(2)}M`
                      : volume >= 1000 
                        ? `${(volume / 1000).toFixed(2)}K`
                        : volume.toLocaleString()
                    }</div>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ 
      background: 'linear-gradient(155deg, #000000ff 0%, #200000ff 25%, #000657ff 100%)',
      minHeight: '100vh',
      padding: '20px',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        background: 'rgba(0, 0, 0, 1)',
        borderRadius: '15px',
        padding: '20px',
        marginBottom: '20px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <h1 style={{ 
              margin: '0 0 5px 0', 
              color : '#2196f3',
              fontSize: '2.5em',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '15px'
            }}>
              <Activity size={40} style={{ color: '#2196f3' }} />
              Multi-Cyrpto Live Chart
            </h1>
            <p style={{ margin: 0, color: '#bbb', fontSize: '1.1em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={18} />
              Real-time price tracking with 1-minute updates
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            {/* Connection Status */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 15px',
              borderRadius: '25px',
              background: connectionStatus === 'connected' ? '#2a2a2a' : '#1a1a1a',
              border: `1px solid ${connectionStatus === 'connected' ? '#4caf50' : '#f44336'}`
            }}>
              {getConnectionIcon()}
              <span style={{ fontSize: '0.9em', fontWeight: 'bold', color: connectionStatus === 'connected' ? '#4caf50' : '#f44336' }}>
                {connectionStatus.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Crypto Stats Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', 
        gap: '15px', 
        marginBottom: '20px' 
      }}>
        {Object.entries(CRYPTO_CONFIG).map(([symbol, config]) => {
          const stats = cryptoStats[symbol];
          const isActive = selectedCryptos[symbol];
          const hasData = stats && stats.available !== false;
          
          // Show all cards, but indicate if data is not available
          const priceChange = stats?.change || 0;
          const isPositive = priceChange >= 0;
          
          return (
            <div
              key={symbol}
              className="crypto-card"
              style={{
                background: 'rgba(0, 0, 0, 1)',
                borderRadius: '15px',
                padding: '20px',
                border: `2px solid ${isActive ? config.color : 'rgba(255, 255, 255, 0.1)'}`,
                transition: 'all 0.3s ease',
                boxShadow: isActive 
                  ? `0 8px 32px ${config.color}60` 
                  : '0 8px 32px rgba(0, 0, 0, 0.3)',
                opacity: hasData ? (isActive ? 1 : 0.6) : 0.3,
                position: 'relative',
                overflow: 'hidden',
                cursor: hasData ? 'pointer' : 'not-allowed',
                backdropFilter: 'blur(10px)'
              }}
            >
              {/* Header with name and toggle button */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                  <div style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    backgroundColor: config.color,
                    marginRight: '10px'
                  }}></div>
                  <h3 style={{ 
                    color: 'white', 
                    margin: 0,
                    fontSize: '16px',
                    fontWeight: 'bold'
                  }}>
                    {config.name}
                  </h3>
                  <span style={{ 
                    color: '#888', 
                    marginLeft: '10px',
                    fontSize: '14px'
                  }}>
                    {config.symbol}
                  </span>
                </div>
                
                {/* ON/OFF Toggle Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (hasData) {
                      toggleCrypto(symbol);
                    }
                  }}
                  disabled={!hasData}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '20px',
                    border: 'none',
                    background: isActive 
                      ? `linear-gradient(135deg, ${config.color}, ${config.color}dd)` 
                      : 'rgba(255, 255, 255, 0.1)',
                    color: 'white',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    cursor: hasData ? 'pointer' : 'not-allowed',
                    transition: 'all 0.3s ease',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    boxShadow: isActive ? `0 2px 10px ${config.color}60` : 'none'
                  }}
                  onMouseEnter={(e) => {
                    if (hasData) {
                      e.target.style.transform = 'scale(1.05)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = 'scale(1)';
                  }}
                >
                  {isActive ? 'ON' : 'OFF'}
                </button>
              </div>
              
              {/* Price and data section */}
              {hasData ? (
                <>
                  <div style={{ 
                    fontSize: '1.8em', 
                    fontWeight: 'bold', 
                    color: '#fff',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    marginBottom: '8px'
                  }}>
                    {formatPrice(stats.price)}
                    {isPositive ? <ArrowUpRight size={20} color="#4caf50" /> : <ArrowDownRight size={20} color="#f44336" />}
                  </div>
                  
                  <div style={{
                    background: isPositive ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)',
                    color: isPositive ? '#4caf50' : '#f44336',
                    padding: '4px 8px',
                    borderRadius: '12px',
                    fontSize: '0.8em',
                    fontWeight: 'bold',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    marginBottom: '10px'
                  }}>
                    {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                    {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
                  </div>
                  
                  <div style={{ 
                    marginTop: '10px', 
                    fontSize: '0.9em', 
                    color: '#bbb',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '5px'
                  }}>
                    <div>Vol: <span style={{ color: '#fff', fontWeight: 'bold' }}>
                      {(stats.volume ?? 0) >= 1000000 
                        ? `${((stats.volume ?? 0) / 1000000).toFixed(2)}M`
                        : (stats.volume ?? 0) >= 1000 
                          ? `${((stats.volume ?? 0) / 1000).toFixed(2)}K`
                          : (stats.volume ?? 0).toLocaleString()
                      }
                    </span></div>
                  </div>
                  
                  {/* Trading Signal Indicator */}
                  <div style={{
                    marginTop: '12px',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    background: tradingSignals[symbol] 
                      ? (tradingSignals[symbol].type === 'BUY' 
                        ? 'rgba(76, 175, 80, 0.3)' 
                        : 'rgba(244, 67, 54, 0.3)')
                      : 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${tradingSignals[symbol] 
                      ? (tradingSignals[symbol].type === 'BUY' ? '#4caf50' : '#f44336')
                      : 'rgba(255, 255, 255, 0.15)'}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '0.9em',
                    fontWeight: 'bold',
                    color: tradingSignals[symbol] 
                      ? (tradingSignals[symbol].type === 'BUY' ? '#4caf50' : '#f44336')
                      : 'rgba(255, 255, 255, 0.5)',
                    opacity: tradingSignals[symbol] ? 1 : 0.5,
                    animation: tradingSignals[symbol] ? 'pulse 2s ease-in-out infinite' : 'none',
                    boxShadow: tradingSignals[symbol]
                      ? (tradingSignals[symbol].type === 'BUY' 
                        ? '0 0 15px rgba(76, 175, 80, 0.6)'
                        : '0 0 15px rgba(244, 67, 54, 0.6)')
                      : 'none',
                    transition: 'all 0.3s ease'
                  }}>
                    {tradingSignals[symbol] ? (
                      tradingSignals[symbol].type === 'BUY' ? (
                        <>
                          <TrendingUp size={16} style={{ color: '#4caf50' }} />
                          <span style={{ color: '#4caf50' }}>BUY Signal</span>
                          <span style={{ fontSize: '0.8em', opacity: 0.9, marginLeft: 'auto' }}>
                            RSI: {tradingSignals[symbol].rsi.toFixed(1)}
                          </span>
                        </>
                      ) : (
                        <>
                          <TrendingDown size={16} style={{ color: '#f44336' }} />
                          <span style={{ color: '#f44336' }}>SELL Signal</span>
                          <span style={{ fontSize: '0.8em', opacity: 0.9, marginLeft: 'auto' }}>
                            RSI: {tradingSignals[symbol].rsi.toFixed(1)}
                          </span>
                        </>
                      )
                    ) : (
                      <>
                        <Minus size={16} />
                        Waiting for signal...
                      </>
                    )}
                  </div>
                </>
              ) : (
                <div style={{ 
                  fontSize: '14px', 
                  color: '#ff6b6b', 
                  marginTop: '10px',
                  fontStyle: 'italic',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <XCircle size={16} />
                  Not available on Alpaca
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Main Chart */}
      <div style={{
        background: 'rgba(0, 0, 0, 1)',
        borderRadius: '15px',
        padding: '20px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '20px',
          gap: '15px',
          flexWrap: 'wrap'
        }}>
          <h3 style={{ 
            margin: '0', 
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '1.3em'
          }}>
            <Activity size={24} />
            XRP Price Movement
            <span style={{ fontSize: '0.7em', color: '#bbb', fontWeight: 'normal' }}>
              ({chartData.length} data points {chartData.length < 5 ? '- accumulating...' : ''})
            </span>
          </h3>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            {/* View Mode Toggle */}
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setViewMode("percentage")}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: '1px solid #555',
                  background: viewMode === "percentage" ? '#444' : '#333',
                  color: '#fff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  fontSize: '0.85em',
                  fontWeight: 'bold',
                  transition: 'all 0.2s ease'
                }}
              >
                <TrendingUp size={14} />
                % Change
              </button>
              <button
                onClick={() => setViewMode("absolute")}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: '1px solid #555',
                  background: viewMode === "absolute" ? '#444' : '#333',
                  color: '#fff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  fontSize: '0.85em',
                  fontWeight: 'bold',
                  transition: 'all 0.2s ease'
                }}
              >
                <DollarSign size={14} />
                $ Price
              </button>
            </div>
            <div style={{
              background: viewMode === 'percentage' ? 'rgba(76, 175, 80, 0.2)' : 'rgba(33, 150, 243, 0.2)',
              color: viewMode === 'percentage' ? '#4caf50' : '#2196f3',
              border: `1px solid ${viewMode === 'percentage' ? '#4caf50' : '#2196f3'}`,
              padding: '8px 15px',
              borderRadius: '15px',
              fontSize: '0.9em',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              {Object.values(selectedCryptos).filter(v => v).length} / {Object.keys(CRYPTO_CONFIG).length} Active
            </div>
          </div>
        </div>
        
        <div style={{ height: 500 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                {/* Create gradients for each crypto */}
                {Object.entries(CRYPTO_CONFIG).map(([symbol, config]) => (
                  <linearGradient key={`gradient-${symbol}`} id={`gradient-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={config.color} stopOpacity={0.6}/>
                    <stop offset="50%" stopColor={config.color} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={config.color} stopOpacity={0}/>
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="displayTime" 
                tick={{ fontSize: 11, fill: '#bbb' }}
                stroke="#555"
                tickLine={false}
              />
              <YAxis 
                tick={{ fontSize: 11, fill: '#bbb' }}
                stroke="#555"
                tickLine={false}
                tickFormatter={(value) => viewMode === "percentage" ? `${value.toFixed(2)}%` : `$${value.toFixed(4)}`}
                domain={viewMode === "percentage" ? ['dataMin - 0.5', 'dataMax + 0.5'] : ['dataMin - 0.002', 'dataMax + 0.002']}
              />
              <Tooltip content={<CustomTooltip />} />
              
              {/* Render area and line for each selected crypto */}
              {Object.entries(CRYPTO_CONFIG).map(([symbol, config]) => (
                selectedCryptos[symbol] && (
                  <React.Fragment key={symbol}>
                    <Area
                      type="monotone"
                      dataKey={viewMode === "percentage" ? `${symbol}_pct` : symbol}
                      fill={`url(#gradient-${symbol})`}
                      stroke="none"
                      isAnimationActive={true}
                      animationDuration={300}
                      connectNulls={true}
                    />
                    <Line
                      type="monotone"
                      dataKey={viewMode === "percentage" ? `${symbol}_pct` : symbol}
                      stroke={config.color}
                      strokeWidth={2}
                      dot={{ r: 4, fill: config.color, stroke: '#fff', strokeWidth: 2 }}
                      name={config.name}
                      isAnimationActive={true}
                      animationDuration={300}
                      connectNulls={true}
                      activeDot={{ 
                        r: 8, 
                        fill: config.color, 
                        stroke: '#fff', 
                        strokeWidth: 2,
                        filter: `drop-shadow(0 0 8px ${config.color})`
                      }}
                      style={{
                        filter: `drop-shadow(0 0 4px ${config.color}60)`
                      }}
                    />
                  </React.Fragment>
                )
              ))}
              
              {/* Add BUY/SELL Signal Markers */}
              {Object.entries(tradingSignals).map(([symbol, signal]) => {
                if (!selectedCryptos[symbol] || !signal || !chartData[signal.index]) return null;
                
                const config = CRYPTO_CONFIG[symbol];
                const dataPoint = chartData[signal.index];
                const value = viewMode === "percentage" ? dataPoint[`${symbol}_pct`] : dataPoint[symbol];
                
                return (
                  <ReferenceLine
                    key={`signal-${symbol}`}
                    x={dataPoint.time}
                    stroke={signal.type === 'BUY' ? '#4caf50' : '#f44336'}
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    label={{
                      value: signal.type === 'BUY' ? '↑ BUY' : '↓ SELL',
                      position: signal.type === 'BUY' ? 'bottom' : 'top',
                      fill: signal.type === 'BUY' ? '#4caf50' : '#f44336',
                      fontSize: 12,
                      fontWeight: 'bold'
                    }}
                  />
                );
              })}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        {/* RSI Chart */}
        <div style={{ marginTop: '30px' }}>
          <h4 style={{ 
            color: '#fff', 
            marginBottom: '15px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '1.1em'
          }}>
            <Activity size={20} />
            RSI (Relative Strength Index)
            <span style={{ fontSize: '0.7em', color: '#bbb', fontWeight: 'normal' }}>
              14-period - Overbought (70) / Oversold (30)
            </span>
          </h4>
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="displayTime" 
                  tick={{ fontSize: 11, fill: '#bbb' }}
                  stroke="#555"
                  tickLine={false}
                />
                <YAxis 
                  tick={{ fontSize: 11, fill: '#bbb' }}
                  stroke="#555"
                  tickLine={false}
                  domain={[0, 100]}
                  label={{ value: 'RSI', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                
                {/* RSI reference lines */}
                <ReferenceLine 
                  y={70} 
                  stroke="#f44336" 
                  strokeDasharray="5 5"
                  label={{ value: 'Overbought (70)', position: 'right', fill: '#f44336', fontSize: 10 }}
                />
                <ReferenceLine 
                  y={30} 
                  stroke="#4caf50" 
                  strokeDasharray="5 5"
                  label={{ value: 'Oversold (30)', position: 'right', fill: '#4caf50', fontSize: 10 }}
                />
                <ReferenceLine 
                  y={50} 
                  stroke="#888" 
                  strokeDasharray="2 2"
                  label={{ value: 'Neutral (50)', position: 'right', fill: '#888', fontSize: 10 }}
                />
                
                {/* RSI Lines for each selected crypto */}
                {Object.entries(CRYPTO_CONFIG).map(([symbol, config]) => {
                  if (!selectedCryptos[symbol]) return null;
                  
                  return (
                    <Line
                      key={`rsi-${symbol}`}
                      type="monotone"
                      dataKey={`${symbol}_rsi`}
                      stroke={config.color}
                      strokeWidth={2.5}
                      dot={false}
                      name={`${config.name} RSI`}
                      isAnimationActive={true}
                      animationDuration={300}
                      connectNulls={true}
                      style={{
                        filter: `drop-shadow(0 0 3px ${config.color}60)`
                      }}
                    />
                  );
                })}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Chart Legend - Single Combined Legend */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            gap: '20px', 
            marginTop: '15px',
            padding: '10px',
            background: '#111',
            borderRadius: '8px',
            flexWrap: 'wrap',
            fontSize: '0.85em'
          }}>
            {Object.entries(CRYPTO_CONFIG).map(([symbol, config]) => (
              selectedCryptos[symbol] && cryptoStats[symbol] && (
                <div key={`legend-${symbol}`} style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#fff' }}>
                  <div style={{ width: '16px', height: '3px', background: config.color, borderRadius: '2px' }}></div>
                  {config.name}
                </div>
              )
            ))}
          </div>
        </div>

        {/* Trading Signals Summary */}
        {Object.keys(tradingSignals).length > 0 && (
          <div style={{
            marginTop: '15px',
            padding: '15px',
            background: 'rgba(0, 0, 0, 0.8)',
            borderRadius: '10px',
            border: '2px solid #444',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '10px'
          }}>
            <div style={{ gridColumn: '1 / -1', fontSize: '1.1em', fontWeight: 'bold', color: '#fff', marginBottom: '10px' }}>
              <Activity size={20} style={{ marginRight: '8px' }} />
              Trading Signals (RSI-Based)
            </div>
            {Object.entries(tradingSignals).map(([symbol, signal]) => {
              const config = CRYPTO_CONFIG[symbol];
              return (
                <div key={symbol} style={{
                  padding: '10px',
                  borderRadius: '8px',
                  background: signal.type === 'BUY' ? 'rgba(76, 175, 80, 0.15)' : 'rgba(244, 67, 54, 0.15)',
                  border: `1px solid ${signal.type === 'BUY' ? '#4caf50' : '#f44336'}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  color: signal.type === 'BUY' ? '#4caf50' : '#f44336',
                  fontWeight: 'bold',
                  fontSize: '0.9em'
                }}>
                  <div style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: config.color
                  }}></div>
                  <div>
                    {config.name}
                    <br/>
                    <span style={{ fontSize: '0.85em', opacity: 0.9 }}>
                      {signal.type} - RSI: {signal.rsi.toFixed(1)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        
        <div style={{ 
          marginTop: '20px', 
          padding: '15px', 
          background: '#222',
          borderRadius: '10px',
          color: '#888',
          fontSize: '13px'
        }}>
          <p style={{ margin: '0 0 5px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} />
            <strong style={{ color: '#fff' }}>Tip:</strong> Use the ON/OFF buttons on each crypto card to show/hide them from the chart
          </p>
          <p style={{ margin: '0 0 5px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={16} />
            <strong style={{ color: '#fff' }}>View Modes:</strong> Toggle between <strong>% Change</strong> (normalized comparison) and <strong>$ Price</strong> (absolute OHLCV data)
          </p>
          <p style={{ margin: '0 0 5px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle size={16} />
            <strong style={{ color: '#fff' }}>Trading Signals:</strong> <span style={{ color: '#4caf50' }}>↑ BUY (RSI &lt; 30 - Oversold)</span> | <span style={{ color: '#f44336' }}>↓ SELL (RSI &gt; 70 - Overbought)</span>
          </p>
          <p style={{ margin: '0 0 5px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <RefreshCw size={16} />
            Chart updates every <strong>1 minute</strong> with live 1-minute bars from Alpaca Markets
          </p>
          
        </div>
      </div>

      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
            100% { opacity: 1; transform: scale(1); }
          }
          
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          
          .animate-spin {
            animation: spin 1s linear infinite;
          }
          
          /* Hover effects for buttons */
          button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.2s ease;
          }
          
          /* Card hover effects */
          .crypto-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
          }
          
          /* Custom scrollbar */
          div::-webkit-scrollbar {
            width: 6px;
            height: 6px;
          }
          
          div::-webkit-scrollbar-track {
            background: #222;
            border-radius: 3px;
          }
          
          div::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 3px;
          }
          
          div::-webkit-scrollbar-thumb:hover {
            background: #666;
          }
        `}
      </style>
    </div>
  );
}
