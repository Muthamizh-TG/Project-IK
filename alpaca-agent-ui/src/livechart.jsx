// src/LiveChart.js
import React, { useEffect, useState, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  AreaChart,
  Area,
  ReferenceLine,
  ComposedChart,
  Bar,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Wifi,
  WifiOff,
  AlertTriangle,
  BarChart3,
  DollarSign,
  Volume2,
  Signal,
  Zap,
  Target,
  Timer,
  Eye,
  Gauge,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Radio,
  Database,
  RefreshCw,
  Play,
  Pause,
  Settings,
  Info,
  Bell,
  CheckCircle,
  XCircle,
  Clock,
} from "lucide-react";

export default function LiveChart() {
  const [points, setPoints] = useState(() => {
    // Load persisted data from localStorage on initialization
    const saved = localStorage.getItem('chartData');
    return saved ? JSON.parse(saved) : [];
  });
  const [alerts, setAlerts] = useState(() => {
    const saved = localStorage.getItem('alertsData');
    return saved ? JSON.parse(saved) : [];
  });

  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [currentPrice, setCurrentPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [volume, setVolume] = useState(0);
  const [rsi, setRsi] = useState(0);
  const [smaShort, setSmaShort] = useState(0);
  const [smaLong, setSmaLong] = useState(0);
  const [chartType, setChartType] = useState("area");
  const [showVolume, setShowVolume] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [historicalContext, setHistoricalContext] = useState({});
  const [showHistoricalPanel, setShowHistoricalPanel] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
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
        if (msg.type === "update") {
          const timeLabel = new Date(msg.timestamp).toLocaleTimeString();
          const price = parseFloat(msg.price);
          const newPoint = {
            time: timeLabel,
            timestamp: msg.timestamp,
            price: price,
            rsi: msg.rsi,
            volume: msg.bar?.v || msg.volume || 0,
            high: msg.bar?.h || price,
            low: msg.bar?.l || price,
            open: msg.bar?.o || price,
            close: msg.bar?.c || price,
          };
          
          setPoints((prev) => {
            const next = [...prev, newPoint].slice(-100);
            // Persist to localStorage
            localStorage.setItem('chartData', JSON.stringify(next));
            return next;
          });
          
          // Calculate price change
          if (points.length > 0) {
            const prevPrice = points[points.length - 1].price;
            const change = ((price - prevPrice) / prevPrice) * 100;
            setPriceChange(change);
          }
          
          setCurrentPrice(price);
          const volumeValue = msg.bar?.v || msg.volume || 0;
          setVolume(volumeValue);
          setRsi(msg.rsi || 0);
          setSmaShort(msg.sma_short || 0);
          setSmaLong(msg.sma_long || 0);
          
          // Update historical context
          if (msg.historical_context) {
            setHistoricalContext(msg.historical_context);
          }
          
          if (msg.alerts && msg.alerts.length) {
            setAlerts((prev) => {
              const next = [{ ts: msg.timestamp, alerts: msg.alerts }, ...prev].slice(0, 20);
              localStorage.setItem('alertsData', JSON.stringify(next));
              return next;
            });
          }
          

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

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case "connected": return "#4caf50";
      case "connecting": return "#ff9800";
      case "disconnected": return "#f44336";
      case "error": return "#f44336";
      default: return "#9e9e9e";
    }
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case "connected": return <Wifi size={16} />;
      case "connecting": return <RefreshCw size={16} className="animate-spin" />;
      case "disconnected": return <WifiOff size={16} />;
      case "error": return <XCircle size={16} />;
      default: return <Radio size={16} />;
    }
  };

  const getPriceIcon = () => {
    if (priceChange > 0) return <ArrowUpRight size={20} color="#4caf50" />;
    if (priceChange < 0) return <ArrowDownRight size={20} color="#f44336" />;
    return <Minus size={20} color="#9e9e9e" />;
  };

  const getRSIIcon = () => {
    if (rsi > 70) return <TrendingUp size={20} color="#f44336" />;
    if (rsi < 30) return <TrendingDown size={20} color="#4caf50" />;
    return <Activity size={20} color="#ff9800" />;
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price);
  };

  const formatPercentage = (value) => {
    const color = value >= 0 ? '#4caf50' : '#f44336';
    const sign = value >= 0 ? '+' : '';
    return <span style={{ color }}>{sign}{value.toFixed(2)}%</span>;
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          background: 'rgba(0, 0, 0, 0.8)',
          padding: '10px',
          borderRadius: '5px',
          color: 'white',
          fontSize: '12px'
        }}>
          <p>{`Time: ${label}`}</p>
          <p>{`Price: ${formatPrice(data.price)}`}</p>
          <p>{`RSI: ${data.rsi?.toFixed(2) || 'N/A'}`}</p>
          <p>{`Volume: ${data.volume?.toLocaleString() || 'N/A'}`}</p>
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
              color : '#ffffff',
              fontSize: '2.5em',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '15px'
            }}>
              <BarChart3 size={40} style={{ color: '#ffffff' }} />
              Project IK  Dashboard
            </h1>
            <p style={{ margin: 0, color: '#bbb', fontSize: '1.1em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Target size={18} />
              Technical Analysis Insights
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            {/* Chart Controls */}
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setChartType(chartType === 'area' ? 'line' : 'area')}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: '1px solid #555',
                  background: '#333',
                  color: '#fff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  fontSize: '0.85em'
                }}
              >
                {chartType === 'area' ? <BarChart3 size={14} /> : <Activity size={14} />}
                {chartType === 'area' ? 'Area' : 'Line'}
              </button>
              <button
                onClick={() => setShowVolume(!showVolume)}
                style={{
                  padding: '8px 12px',
                  borderRadius: '8px',
                  border: '1px solid #555',
                  background: showVolume ? '#444' : '#333',
                  color: '#fff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  fontSize: '0.85em'
                }}
              >
                <Volume2 size={14} />
                Volume
              </button>
            </div>
            
            {/* Connection Status */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 15px',
              borderRadius: '25px',
              background: connectionStatus === 'connected' ? '#2a2a2a' : '#1a1a1a',
              border: `1px solid ${getConnectionStatusColor()}`
            }}>
              {getConnectionIcon()}
              <span style={{ fontSize: '0.9em', fontWeight: 'bold', color: getConnectionStatusColor() }}>
                {connectionStatus.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Market Stats Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', 
        gap: '15px', 
        marginBottom: '20px' 
      }}>
        <div style={{
          background: 'rgba(0, 0, 0, 1)',
          borderRadius: '15px',
          padding: '20px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden',
          cursor: 'pointer'
        }}
        className="market-card">
          <div style={{ 
            position: 'absolute', 
            top: '15px', 
            right: '15px',
            opacity: 0.2
          }}>
          </div>
          <div style={{ 
            fontSize: '0.9em', 
            color: '#bbb', 
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Target size={16} />
            Current Price
          </div>
          <div style={{ 
            fontSize: '1.8em', 
            fontWeight: 'bold', 
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            {formatPrice(currentPrice)}
            {getPriceIcon()}
          </div>
          <div style={{ fontSize: '0.9em', marginTop: '8px' }}>
            {formatPercentage(priceChange)}
          </div>
        </div>

        <div style={{
          background: 'rgba(0, 0, 0, 1)',
          borderRadius: '15px',
          padding: '20px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ 
            position: 'absolute', 
            top: '15px', 
            right: '15px',
            opacity: 0.2
          }}>
          </div>
          <div style={{ 
            fontSize: '0.9em', 
            color: '#bbb', 
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Activity size={16} />
            RSI (14)
          </div>
          <div style={{ 
            fontSize: '1.8em', 
            fontWeight: 'bold', 
            color: rsi > 70 ? '#f44336' : rsi < 30 ? '#4caf50' : '#fff',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            {rsi ? rsi.toFixed(1) : 'N/A'}
            {getRSIIcon()}
          </div>
          <div style={{ fontSize: '0.9em', marginTop: '8px', color: '#bbb' }}>
            {rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral'}
          </div>
        </div>

        <div style={{
          background: 'rgba(0, 0, 0, 1)',
          borderRadius: '15px',
          padding: '20px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ 
            position: 'absolute', 
            top: '15px', 
            right: '15px',
            opacity: 0.2
          }}>
          </div>
          <div style={{ 
            fontSize: '0.9em', 
            color: '#bbb', 
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <BarChart3 size={16} />
            Volume
          </div>
          <div style={{ 
            fontSize: '1.8em', 
            fontWeight: 'bold', 
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            {volume !== null && volume !== undefined && volume > 0 
              ? volume >= 1000000 
                ? `${(volume / 1000000).toFixed(2)}M`
                : volume >= 1000 
                  ? `${(volume / 1000).toFixed(2)}K`
                  : volume.toLocaleString()
              : 'No Data'}
            <Signal size={20} color="#fff" />
          </div>
          <div style={{ fontSize: '0.9em', marginTop: '8px', color: '#bbb' }}>24h Volume</div>
        </div>

        <div style={{
          background: 'rgba(0, 0, 0, 1)',
          borderRadius: '15px',
          padding: '20px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ 
            position: 'absolute', 
            top: '15px', 
            right: '15px',
            opacity: 0.2
          }}>
          </div>
          <div style={{ 
            fontSize: '0.9em', 
            color: '#bbb', 
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Eye size={16} />
            Data Points
          </div>
          <div style={{ 
            fontSize: '1.8em', 
            fontWeight: 'bold', 
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            {points.length}
            <CheckCircle size={20} color="#4caf50" />
          </div>
          <div style={{ fontSize: '0.9em', marginTop: '8px', color: '#bbb' }}>Last 100 updates</div>
        </div>

        {/* Additional SMA Cards */}
        <div style={{
          background: 'rgba(0, 0, 0, 1)',
          borderRadius: '15px',
          padding: '20px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ 
            position: 'absolute', 
            top: '15px', 
            right: '15px',
            opacity: 0.2
          }}>
          </div>
          <div style={{ 
            fontSize: '0.9em', 
            color: '#bbb', 
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Zap size={16} />
            SMA 50
          </div>
          <div style={{ 
            fontSize: '1.8em', 
            fontWeight: 'bold', 
            color: '#ccc',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            {smaShort ? formatPrice(smaShort) : 'N/A'}
          </div>
          <div style={{ fontSize: '0.9em', marginTop: '8px', color: '#bbb' }}>Short Moving Avg</div>
        </div>

        <div style={{
          background: 'rgba(0, 0, 0, 1)',
          borderRadius: '15px',
          padding: '20px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ 
            position: 'absolute', 
            top: '15px', 
            right: '15px',
            opacity: 0.2
          }}>
          </div>
          <div style={{ 
            fontSize: '0.9em', 
            color: '#bbb', 
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Activity size={16} />
            SMA 200
          </div>
          <div style={{ 
            fontSize: '1.8em', 
            fontWeight: 'bold', 
            color: '#ccc',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            {smaLong ? formatPrice(smaLong) : 'N/A'}
          </div>
          <div style={{ fontSize: '0.9em', marginTop: '8px', color: '#bbb' }}>Long Moving Avg</div>
        </div>

        
      </div>

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        
        {/* Charts Section */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Price Chart */}
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
              marginBottom: '20px' 
            }}>
              <h3 style={{ 
                margin: '0', 
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <BarChart3 size={24} />
                Price Chart
                <span style={{ fontSize: '0.8em', color: '#bbb', fontWeight: 'normal' }}>
                  ({points.length} points)
                </span>
              </h3>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '5px',
                  fontSize: '0.85em',
                  color: '#bbb'
                }}>
                  <Clock size={14} />
                  Live Updates
                </div>
                {currentPrice > 0 && (
                  <div style={{
                    background: priceChange >= 0 ? '#e8f5e8' : '#ffebee',
                    color: priceChange >= 0 ? '#2e7d32' : '#c62828',
                    padding: '4px 8px',
                    borderRadius: '12px',
                    fontSize: '0.8em',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    {priceChange >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                    {Math.abs(priceChange).toFixed(2)}%
                  </div>
                )}
              </div>
            </div>
            <div style={{ height: 450 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={points}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffffff" stopOpacity={0.4}/>
                      <stop offset="50%" stopColor="#ffffff" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#ffffff" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#888888" stopOpacity={0.6}/>
                      <stop offset="95%" stopColor="#888888" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="2 2" stroke="#333" />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fontSize: 11, fill: '#bbb' }}
                    stroke="#555"
                    tickLine={false}
                  />
                  <YAxis 
                    yAxisId="price"
                    domain={["dataMin - 5", "dataMax + 5"]} 
                    tick={{ fontSize: 11, fill: '#bbb' }}
                    stroke="#555"
                    tickLine={false}
                    tickFormatter={(value) => `$${value.toFixed(2)}`}
                  />
                  {showVolume && (
                    <YAxis 
                      yAxisId="volume"
                      orientation="right"
                      tick={{ fontSize: 11, fill: '#bbb' }}
                      stroke="#555"
                      tickLine={false}
                      tickFormatter={(value) => `${(value / 1000).toFixed(1)}K`}
                    />
                  )}
                  <Tooltip content={<CustomTooltip />} />
                  
                  {chartType === 'area' ? (
                    <Area
                      yAxisId="price"
                      type="monotone"
                      dataKey="price"
                      stroke="#ffffff"
                      strokeWidth={3}
                      fill="url(#priceGradient)"
                      dot={false}
                      activeDot={{ 
                        r: 8, 
                        fill: "#ffffff", 
                        stroke: '#333', 
                        strokeWidth: 2,
                        filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))'
                      }}
                    />
                  ) : (
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="price"
                      stroke="#ffffff"
                      strokeWidth={3}
                      dot={false}
                      activeDot={{ 
                        r: 8, 
                        fill: "#ffffff", 
                        stroke: '#333', 
                        strokeWidth: 2 
                      }}
                    />
                  )}
                  
                  {showVolume && (
                    <Bar 
                      yAxisId="volume"
                      dataKey="volume" 
                      fill="url(#volumeGradient)"
                      opacity={0.6}
                      radius={[2, 2, 0, 0]}
                    />
                  )}
                  
                  {/* Moving Averages */}
                  {smaShort > 0 && (
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey={() => smaShort}
                      stroke="#ccc"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      connectNulls={false}
                    />
                  )}
                  {smaLong > 0 && (
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey={() => smaLong}
                      stroke="#999"
                      strokeWidth={2}
                      strokeDasharray="10 5"
                      dot={false}
                      connectNulls={false}
                    />
                  )}
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
              borderRadius: '8px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                <div style={{ width: '16px', height: '3px', background: '#ffffff', borderRadius: '2px' }}></div>
                Price
              </div>
              {showVolume && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                  <div style={{ width: '16px', height: '8px', background: '#888888', borderRadius: '2px', opacity: 0.6 }}></div>
                  Volume
                </div>
              )}
              {smaShort > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                  <div style={{ width: '16px', height: '2px', background: '#ccc', borderRadius: '1px', border: '1px dashed #ccc' }}></div>
                  SMA 50
                </div>
              )}
              {smaLong > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                  <div style={{ width: '16px', height: '2px', background: '#999', borderRadius: '1px', border: '1px dashed #999' }}></div>
                  SMA 200
                </div>
              )}
            </div>
          </div>

          {/* RSI Chart */}
          <div style={{
            background: 'rgba(0, 0, 0, 1)',
            borderRadius: '15px',
            padding: '20px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              marginBottom: '20px' 
            }}>
              <h3 style={{ 
                margin: '0', 
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <Gauge size={24} />
                RSI Indicator
              </h3>
              <div style={{
                background: rsi > 70 ? 'rgba(244, 67, 54, 0.2)' : rsi < 30 ? 'rgba(76, 175, 80, 0.2)' : 'rgba(158, 158, 158, 0.2)',
                color: rsi > 70 ? '#f44336' : rsi < 30 ? '#4caf50' : '#9e9e9e',
                border: `1px solid ${rsi > 70 ? '#f44336' : rsi < 30 ? '#4caf50' : '#9e9e9e'}`,
                padding: '8px 15px',
                borderRadius: '15px',
                fontSize: '0.9em',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                {getRSIIcon()}
                <span>RSI: {rsi ? rsi.toFixed(1) : 'N/A'}</span>
                <span>({rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral'})</span>
              </div>
            </div>
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={points.filter(point => point.rsi !== null && point.rsi !== undefined)}>
                  <defs>
                    <linearGradient id="rsiGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#4fc3f7" stopOpacity={0.6}/>
                      <stop offset="30%" stopColor="#81c784" stopOpacity={0.4}/>
                      <stop offset="70%" stopColor="#ffb74d" stopOpacity={0.4}/>
                      <stop offset="100%" stopColor="#f48fb1" stopOpacity={0.6}/>
                    </linearGradient>
                    <linearGradient id="rsiLineGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f44336" stopOpacity={1}/>
                      <stop offset="30%" stopColor="#ffffff" stopOpacity={1}/>
                      <stop offset="70%" stopColor="#ffffff" stopOpacity={1}/>
                      <stop offset="100%" stopColor="#4caf50" stopOpacity={1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" strokeOpacity={0.5} />
                  <XAxis 
                    dataKey="time" 
                    tick={{ fontSize: 10, fill: '#bbb' }} 
                    stroke="#666"
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    domain={[0, 100]} 
                    tick={{ fontSize: 10, fill: '#bbb' }} 
                    stroke="#666"
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${value}`}
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      value ? `${value.toFixed(2)}` : 'N/A', 
                      'RSI (14)'
                    ]}
                    labelFormatter={(label) => `Time: ${label}`}
                    labelStyle={{ color: '#fff', fontWeight: 'bold' }}
                    contentStyle={{
                      background: 'rgba(20, 20, 20, 0.95)',
                      border: '1px solid #666',
                      borderRadius: '8px',
                      color: 'white',
                      fontSize: '0.9em'
                    }}
                  />
                  
                  {/* Overbought Zone (70-100) */}
                  <ReferenceLine 
                    y={70} 
                    stroke="#f44336" 
                    strokeDasharray="8 4" 
                    strokeWidth={2}
                    strokeOpacity={0.8}
                  />
                  
                  {/* Oversold Zone (0-30) */}
                  <ReferenceLine 
                    y={30} 
                    stroke="#4caf50" 
                    strokeDasharray="8 4" 
                    strokeWidth={2}
                    strokeOpacity={0.8}
                  />
                  
                  {/* Midline (50) */}
                  <ReferenceLine 
                    y={50} 
                    stroke="#9e9e9e" 
                    strokeDasharray="2 2" 
                    strokeWidth={1}
                    strokeOpacity={0.6}
                  />
                  
                  {/* RSI Area */}
                  <Area
                    type="monotone"
                    dataKey="rsi"
                    stroke="url(#rsiLineGradient)"
                    strokeWidth={2.5}
                    fill="url(#rsiGradient)"
                    fillOpacity={0.3}
                    dot={false}
                    activeDot={{ 
                      r: 5, 
                      fill: "#ffffff", 
                      stroke: '#333', 
                      strokeWidth: 2,
                      filter: 'drop-shadow(0 0 4px rgba(255,255,255,0.3))'
                    }}
                    connectNulls={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            {/* RSI Legend */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'center', 
              gap: '20px', 
              marginTop: '15px',
              padding: '10px',
              background: '#222',
              borderRadius: '8px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                <div style={{ width: '16px', height: '3px', background: '#4caf50', borderRadius: '2px' }}></div>
                <span>Oversold (&lt;30)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                <div style={{ width: '16px', height: '3px', background: '#ffffff', borderRadius: '2px' }}></div>
                <span>Current: {rsi?.toFixed(1) || 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#fff' }}>
                <div style={{ width: '16px', height: '3px', background: '#f44336', borderRadius: '2px' }}></div>
                <span>Overbought (&gt;70)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Alerts */}
          <div style={{
            background: 'rgba(0, 0, 0, 1)',
            borderRadius: '15px',
            padding: '20px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            maxHeight: '650px',
            overflowY: 'auto',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{ 
              position: 'absolute', 
              top: '15px', 
              right: '15px',
              opacity: 0.1
            }}>
            </div>
            <h3 style={{ 
              margin: '0 0 15px 0', 
              color: '#fff',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <Bell size={24} color="#ffffff" />
              Trading Alerts
              {alerts.length > 0 && (
                <div style={{
                  background: 'linear-gradient(45deg, #f44336, #d32f2f)',
                  color: 'white',
                  borderRadius: '10%',
                  width: '25px',
                  height: '25px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.75em',
                  fontWeight: 'bold',
                  boxShadow: '0 2px 8px rgba(244, 67, 54, 0.3)',
                  animation: alerts.length > 0 ? 'pulse 2s infinite' : 'none'
                }}>
                  {alerts.length}
                </div>
              )}
            </h3>
            
            <div style={{ overflowY: 'auto', maxHeight: '550px' }}>
              {alerts.length === 0 ? (
                <div style={{ 
                  padding: '30px 20px', 
                  textAlign: 'center', 
                  color: '#ccc',
                  fontSize: '0.9em'
                }}>
                  <div style={{ marginBottom: '15px' }}>
                    <CheckCircle size={32} color="#ffffff" />
                  </div>
                  <div style={{ fontWeight: '500', marginBottom: '5px', color: '#fff' }}>
                    All Clear!
                  </div>
                  <div style={{ fontSize: '0.85em', color: '#ccc' }}>
                    No trading alerts detected
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {alerts.map((a, idx) => (
                    <div key={idx} style={{
                      background: 'rgba(30, 30, 30, 0.8)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '12px',
                      padding: '15px',
                      fontSize: '0.85em',
                      position: 'relative',
                      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
                    }}>
                      <div style={{
                        position: 'absolute',
                        top: '8px',
                        right: '8px',
                        background: '#ffffff',
                        borderRadius: '50%',
                        width: '6px',
                        height: '6px'
                      }}></div>
                      
                      <div style={{ 
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        fontWeight: 'bold', 
                        color: '#fff',
                        marginBottom: '8px',
                        fontSize: '0.8em'
                      }}>
                        <Clock size={14} color="#fff" />
                        {new Date(a.ts).toLocaleString()}
                      </div>
                      
                      {a.alerts.map((alert, i) => (
                        <div key={i} style={{ 
                          color: '#ccc',
                          padding: '4px 0',
                          fontWeight: '500',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}>
                          <AlertTriangle size={16} color="#ffffff" />
                          {alert}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Historical Analysis Panel */}
      {showHistoricalPanel && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(5px)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          animation: 'fadeIn 0.3s ease-out'
        }}
        onClick={() => setShowHistoricalPanel(false)}
        >
          <div style={{
            background: 'rgba(30, 30, 30, 0.98)',
            borderRadius: '20px',
            padding: '30px',
            maxWidth: '800px',
            maxHeight: '80vh',
            overflowY: 'auto',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
            animation: 'slideIn 0.3s ease-out'
          }}
          onClick={(e) => e.stopPropagation()}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '25px'
            }}>
              <h2 style={{
                margin: 0,
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <Database size={28} />
                Historical Analysis & Pattern Intelligence
              </h2>
              <button
                onClick={() => setShowHistoricalPanel(false)}
                style={{
                  background: 'transparent',
                  border: '2px solid #555',
                  borderRadius: '50%',
                  width: '40px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: '#fff'
                }}
              >
                <XCircle size={20} />
              </button>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '25px'
            }}>
              <div style={{
                background: 'rgba(20, 20, 20, 0.8)',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h3 style={{ color: '#fff', margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Bell size={20} />
                  Recent Activity
                </h3>
                <div style={{ color: '#ccc', fontSize: '0.9em' }}>
                  <div style={{ marginBottom: '8px' }}>
                    Alerts (24h): <span style={{ color: '#fff', fontWeight: 'bold' }}>{historicalContext.alerts_24h || 0}</span>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    Patterns (7d): <span style={{ color: '#fff', fontWeight: 'bold' }}>{historicalContext.patterns_7d || 0}</span>
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#888', marginTop: '12px' }}>
                    Data stored in MongoDB for advanced pattern analysis
                  </div>
                </div>
              </div>

              <div style={{
                background: 'rgba(20, 20, 20, 0.8)',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h3 style={{ color: '#fff', margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <TrendingUp size={20} />
                  AI Enhancement
                </h3>
                <div style={{ color: '#ccc', fontSize: '0.9em' }}>
                  <div style={{ marginBottom: '8px' }}>
                    ✅ Historical pattern success rates
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    ✅ Trend strength analysis
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    ✅ Alert correlation tracking
                  </div>
                  <div style={{ fontSize: '0.8em', color: '#888', marginTop: '12px' }}>
                    Powered by Ollama AI with MongoDB context
                  </div>
                </div>
              </div>

              <div style={{
                background: 'rgba(20, 20, 20, 0.8)',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h3 style={{ color: '#fff', margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BarChart3 size={20} />
                  Data Storage
                </h3>
                <div style={{ color: '#ccc', fontSize: '0.9em' }}>
                  <div style={{ marginBottom: '8px' }}>
                    📊 Market data with indicators
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    🚨 Alerts with metadata
                  </div>

                  <div style={{ marginBottom: '8px' }}>
                    📈 Detected patterns
                  </div>
                </div>
              </div>
            </div>

            <div style={{
              background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
              padding: '20px',
              borderRadius: '12px',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}>
              <h3 style={{ color: '#fff', margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Brain size={20} />
                Enhanced Features Now Active
              </h3>
              <div style={{ color: '#ccc', fontSize: '0.9em', lineHeight: '1.6' }}>
                Your trading dashboard now includes advanced MongoDB integration for:
                <br />• <strong>Pattern Success Tracking:</strong> Historical accuracy rates for Golden Cross, Death Cross, RSI signals, and candlestick patterns
                <br />• <strong>Trend Analysis:</strong> 24-hour trend strength calculation using statistical regression
                <br />• <strong>AI Context:</strong> Enhanced analysis considering historical performance and market correlations
                <br />• <strong>Alert Intelligence:</strong> Smart alerting with cooldown periods and historical context
                <br />• <strong>Data Persistence:</strong> All market data, alerts, and AI insights stored for future analysis
              </div>
            </div>
          </div>
        </div>
      )}

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
          
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          
          @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
          }
          
          .animate-spin {
            animation: spin 1s linear infinite;
          }
          
          .animate-fadeIn {
            animation: fadeIn 0.5s ease-out;
          }
          
          .animate-slideIn {
            animation: slideIn 0.3s ease-out;
          }
          
          /* Custom scrollbar for alerts */
          div::-webkit-scrollbar {
            width: 6px;
          }
          
          div::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
          }
          
          div::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
          }
          
          div::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
          }
          
          /* Hover effects for buttons */
          button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.2s ease;
          }
          
          /* Card hover effects */
          .market-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
          }
        `}
      </style>
    </div>
  );
}
