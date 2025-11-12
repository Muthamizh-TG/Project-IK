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
} from "lucide-react";

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
    'BTC/USD': false,
    'ETH/USD': false,
    'SOL/USD': false,
    'XRP/USD': true,
    'DOGE/USD': true,
    'HYPER/USD': true,
    'PEPENODE/USD': true,
    'BEST/USD': true,
    'LILPEPE/USD': true,
    'BONK/USD': true,
  });
  const wsRef = useRef(null);

  // Crypto configuration with colors - XRP focused
  const CRYPTO_CONFIG = {
    'BTC/USD': { color: '#F7931A', name: 'Bitcoin', symbol: 'BTC' },
    'ETH/USD': { color: '#627EEA', name: 'Ethereum', symbol: 'ETH' },
    'SOL/USD': { color: '#14F195', name: 'Solana', symbol: 'SOL' },
    'XRP/USD': { color: '#2196f3', name: 'Ripple', symbol: 'XRP' },
    'DOGE/USD': { color: '#C2A633', name: 'Dogecoin', symbol: 'DOGE' },
    'HYPER/USD': { color: '#FF6B6B', name: 'Hyper', symbol: 'HYPER' },
    'PEPENODE/USD': { color: '#4ECDC4', name: 'PepeNode', symbol: 'PEPENODE' },
    'BEST/USD': { color: '#95E1D3', name: 'Best', symbol: 'BEST' },
    'LILPEPE/USD': { color: '#F38181', name: 'LilPepe', symbol: 'LILPEPE' },
    'BONK/USD': { color: '#AA96DA', name: 'Bonk', symbol: 'BONK' },
  };

  useEffect(() => {
    const wsUrl = `ws://localhost:8000/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
      setConnectionStatus("connected");
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        
        if (msg.type === "historical_data") {
          // Process historical data on initial connection
          console.log("Received historical data:", msg);
          
          setBaselinePrices((prevBaseline) => {
            const newBaseline = { ...prevBaseline };
            const historyPoints = [];
            
            // Process each cryptocurrency's historical bars
            Object.entries(msg.data).forEach(([symbol, data]) => {
              if (data.bars && data.bars.length > 0) {
                // Set baseline from first historical bar
                if (!newBaseline[symbol]) {
                  newBaseline[symbol] = data.bars[0].price;
                }
                
                // Add all historical bars to chart
                data.bars.forEach((bar, idx) => {
                  if (!historyPoints[idx]) {
                    historyPoints[idx] = {
                      time: bar.time,
                      timestamp: bar.timestamp
                    };
                  }
                  
                  historyPoints[idx][symbol] = bar.close;
                  historyPoints[idx][`${symbol}_close`] = bar.close;
                  historyPoints[idx][`${symbol}_open`] = bar.open;
                  historyPoints[idx][`${symbol}_high`] = bar.high;
                  historyPoints[idx][`${symbol}_low`] = bar.low;
                  historyPoints[idx][`${symbol}_volume`] = bar.volume;
                  
                  // Calculate percentage change from baseline
                  if (newBaseline[symbol]) {
                    historyPoints[idx][`${symbol}_pct`] = ((bar.close - newBaseline[symbol]) / newBaseline[symbol]) * 100;
                  } else {
                    historyPoints[idx][`${symbol}_pct`] = 0;
                  }
                });
              }
            });
            
            // Set historical data in chart
            setChartData(historyPoints.filter(p => p)); // Remove undefined entries
            localStorage.setItem('multiCryptoChartData', JSON.stringify(historyPoints.filter(p => p)));
            
            // Update crypto stats from historical data
            const stats = {};
            Object.entries(msg.data).forEach(([symbol, data]) => {
              stats[symbol] = data;
            });
            setCryptoStats(stats);
            
            localStorage.setItem('multiCryptoBaselines', JSON.stringify(newBaseline));
            return newBaseline;
          });
          
        } else if (msg.type === "multi_crypto_update") {
          // Process new update data (every 5 minutes)
          const timestamp = new Date(msg.timestamp);
          const timeLabel = timestamp.toLocaleTimeString();
          
          console.log("Received crypto update:", msg);
          
          // Update baseline and chart data together
          setBaselinePrices((prevBaseline) => {
            const newBaseline = { ...prevBaseline };
            
            // Set baseline for new cryptos
            Object.entries(msg.data).forEach(([symbol, data]) => {
              if (!newBaseline[symbol]) {
                newBaseline[symbol] = data.price;
              }
            });
            
            // Create new data point with prices, percentages, and OHLCV data
            const newDataPoint = {
              time: timeLabel,
              timestamp: msg.timestamp,
            };
            
            Object.entries(msg.data).forEach(([symbol, data]) => {
              // Price data
              newDataPoint[symbol] = data.price;
              newDataPoint[`${symbol}_close`] = data.close;
              newDataPoint[`${symbol}_open`] = data.open;
              newDataPoint[`${symbol}_high`] = data.high;
              newDataPoint[`${symbol}_low`] = data.low;
              newDataPoint[`${symbol}_volume`] = data.volume;
              
              // Calculate percentage change from baseline
              if (newBaseline[symbol]) {
                newDataPoint[`${symbol}_pct`] = ((data.price - newBaseline[symbol]) / newBaseline[symbol]) * 100;
              } else {
                newDataPoint[`${symbol}_pct`] = 0;
              }
            });
            
            // Update chart data (keep last 100 points for better visibility)
            setChartData((prevData) => {
              const next = [...prevData, newDataPoint].slice(-100);
              // Persist to localStorage
              localStorage.setItem('multiCryptoChartData', JSON.stringify(next));
              return next;
            });
            
            // Persist baselines to localStorage
            localStorage.setItem('multiCryptoBaselines', JSON.stringify(newBaseline));
            
            return newBaseline;
          });
          
          // Update crypto statistics
          setCryptoStats(msg.data);
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
  }, []);

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

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      // Get the data point for this timestamp
      const dataPoint = chartData.find(d => d.time === label);
      
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
          {payload.map((entry, index) => {
            // Extract the actual symbol (remove _pct suffix if present)
            const symbol = entry.dataKey.replace('_pct', '');
            const config = CRYPTO_CONFIG[symbol];
            if (!config) return null;
            
            // Get OHLCV data if available
            const price = dataPoint ? dataPoint[symbol] : null;
            const open = dataPoint ? dataPoint[`${symbol}_open`] : null;
            const high = dataPoint ? dataPoint[`${symbol}_high`] : null;
            const low = dataPoint ? dataPoint[`${symbol}_low`] : null;
            const volume = dataPoint ? dataPoint[`${symbol}_volume`] : null;
            
            return (
              <div key={index} style={{ 
                marginBottom: '8px',
                paddingBottom: '8px',
                borderBottom: index < payload.length - 1 ? '1px solid #222' : 'none'
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
              Live Chart
            </h1>
            <p style={{ margin: 0, color: '#bbb', fontSize: '1.1em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={18} />
              Real-time price tracking with 5-minute updates
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
                      {stats.volume >= 1000000 
                        ? `${(stats.volume / 1000000).toFixed(2)}M`
                        : stats.volume >= 1000 
                          ? `${(stats.volume / 1000).toFixed(2)}K`
                          : stats.volume.toLocaleString()
                      }
                    </span></div>
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
                dataKey="time" 
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
              <Legend 
                wrapperStyle={{ paddingTop: '20px' }}
                iconType="circle"
              />
              
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
                      strokeWidth={3}
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
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        {/* Chart Legend */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          gap: '20px', 
          marginTop: '15px',
          padding: '10px',
          background: '#222',
          borderRadius: '8px',
          flexWrap: 'wrap'
        }}>
          {Object.entries(CRYPTO_CONFIG).map(([symbol, config]) => (
            selectedCryptos[symbol] && cryptoStats[symbol] && (
              <div key={symbol} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                <div style={{ width: '16px', height: '3px', background: config.color, borderRadius: '2px' }}></div>
                {config.name}
              </div>
            )
          ))}
        </div>
        
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
            <RefreshCw size={16} />
            Chart updates every <strong>5 minute</strong> with live 5-minute bars from Alpaca Markets
          </p>
          <p style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <XCircle size={16} />
            Some tokens may not be available on Alpaca (only major cryptos are supported)
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
