// Simple Crypto Agent Dashboard - WebSocket Version
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
  Signal,
  Zap,
  Target,
  Timer,
  Eye,
  Bell,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
} from "lucide-react";

export default function CryptoDashboard() {
  // State management
  const [marketData, setMarketData] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [lastUpdate, setLastUpdate] = useState("");
  const [serverStats, setServerStats] = useState({});
  const wsRef = useRef(null);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = `${protocol}://${window.location.hostname}:8001/ws`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected to crypto agent");
        setConnectionStatus("connected");
        
        // Send ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 30000);
        
        ws.pingInterval = pingInterval;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case "market_data":
              setMarketData(message.data);
              setLastUpdate(message.timestamp);
              break;
              
            case "alert":
              setAlerts(prev => [message.data, ...prev].slice(0, 50)); // Keep last 50 alerts
              break;
              
            case "pong":
              // Connection health check response
              break;
              
            default:
              console.log("Unknown message type:", message.type);
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket disconnected");
        setConnectionStatus("disconnected");
        if (ws.pingInterval) {
          clearInterval(ws.pingInterval);
        }
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnectionStatus("error");
      };
    };

    connectWebSocket();
    
    // Fetch initial server status
    fetchServerStatus();

    return () => {
      if (wsRef.current) {
        if (wsRef.current.pingInterval) {
          clearInterval(wsRef.current.pingInterval);
        }
        wsRef.current.close();
      }
    };
  }, []);

  const fetchServerStatus = async () => {
    try {
      const response = await fetch(`http://localhost:8001/status`);
      const data = await response.json();
      setServerStats(data);
    } catch (error) {
      console.error("Error fetching server status:", error);
    }
  };

  // Helper functions
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case "connected": return "#4caf50";
      case "connecting": return "#ff9800";
      case "disconnected": return "#f44336";
      case "error": return "#f44336";
      default: return "#999";
    }
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case "connected": return <Wifi size={16} color="#4caf50" />;
      case "connecting": return <RefreshCw size={16} color="#ff9800" className="animate-spin" />;
      default: return <WifiOff size={16} color="#f44336" />;
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case "bullish": return <TrendingUp size={16} color="#4caf50" />;
      case "bearish": return <TrendingDown size={16} color="#f44336" />;
      default: return <Activity size={16} color="#999" />;
    }
  };

  const formatPrice = (price) => {
    if (price === undefined || price === null) return "N/A";
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return "N/A";
    return new Date(timestamp).toLocaleTimeString();
  };

  const getPatternColor = (pattern) => {
    return pattern === "Golden Cross" ? "#4caf50" : "#f44336";
  };

  const getPatternIcon = (pattern) => {
    return pattern === "Golden Cross" ? 
      <TrendingUp size={16} color="#4caf50" /> : 
      <TrendingDown size={16} color="#f44336" />;
  };

  return (
    <div style={{ 
      background: 'linear-gradient(155deg, #000000ff 0%, #1a1a2e 50%, #16213e 100%)',
      minHeight: '100vh',
      padding: '20px',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
      color: '#fff'
    }}>
      {/* Header */}
      <div style={{
        background: 'rgba(30, 30, 30, 0.95)',
        borderRadius: '15px',
        padding: '20px',
        marginBottom: '20px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ 
              margin: '0 0 5px 0', 
              color: '#ffffff',
              fontSize: '2.5em',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '15px'
            }}>
              <BarChart3 size={40} style={{ color: '#ffffff' }} />
              Project IK Dashboard (Simple)
            </h1>
            <p style={{ margin: 0, color: '#bbb', fontSize: '1.1em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Target size={18} />
              Real-time Golden Cross & Death Cross Detection
            </p>
          </div>
          
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

      {/* Market Data Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
        gap: '20px', 
        marginBottom: '30px' 
      }}>
        {Object.entries(marketData).map(([symbol, data]) => (
          <div key={symbol} style={{
            background: 'rgba(30, 30, 30, 0.95)',
            borderRadius: '15px',
            padding: '25px',
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
              <DollarSign size={30} color="#666" />
            </div>
            
            <div style={{ 
              fontSize: '1.2em', 
              fontWeight: 'bold',
              color: '#fff', 
              marginBottom: '15px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              {getTrendIcon(data.trend)}
              {symbol}
            </div>
            
            <div style={{ 
              fontSize: '2.2em', 
              fontWeight: 'bold', 
              color: '#fff',
              marginBottom: '15px'
            }}>
              {formatPrice(data.price)}
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              <div>
                <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>SMA 50</div>
                <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
                  {formatPrice(data.sma_short)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>SMA 200</div>
                <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
                  {formatPrice(data.sma_long)}
                </div>
              </div>
            </div>
            
            <div style={{ 
              marginTop: '15px', 
              padding: '10px', 
              borderRadius: '8px', 
              background: data.trend === 'bullish' ? 'rgba(76, 175, 80, 0.1)' : 
                         data.trend === 'bearish' ? 'rgba(244, 67, 54, 0.1)' : 
                         'rgba(158, 158, 158, 0.1)',
              border: `1px solid ${data.trend === 'bullish' ? '#4caf50' : 
                                  data.trend === 'bearish' ? '#f44336' : '#999'}`
            }}>
              <div style={{ 
                fontSize: '0.9em', 
                fontWeight: 'bold',
                color: data.trend === 'bullish' ? '#4caf50' : 
                       data.trend === 'bearish' ? '#f44336' : '#999',
                textTransform: 'capitalize'
              }}>
                {data.trend} Trend
              </div>
              <div style={{ fontSize: '0.8em', color: '#bbb', marginTop: '5px' }}>
                Last updated: {formatTime(data.last_updated)}
              </div>
              {data.ai_analysis && (
                <div style={{ 
                  fontSize: '0.75em', 
                  color: '#e0e0e0', 
                  marginTop: '8px',
                  padding: '8px',
                  background: 'rgba(0, 0, 0, 0.3)',
                  borderRadius: '5px',
                  borderLeft: '3px solid #2196f3'
                }}>
                  <div style={{ color: '#2196f3', fontWeight: 'bold', marginBottom: '3px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <Target size={12} />
                    Technical Analysis:
                  </div>
                  {data.ai_analysis}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Alerts Section */}
      <div style={{
        background: 'rgba(30, 30, 30, 0.95)',
        borderRadius: '15px',
        padding: '25px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        marginBottom: '30px'
      }}>
        <h2 style={{ 
          margin: '0 0 20px 0', 
          color: '#fff',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <Bell size={24} />
          Trading Alerts
          {alerts.length > 0 && (
            <span style={{
              background: '#f44336',
              color: 'white',
              borderRadius: '50%',
              width: '24px',
              height: '24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.8em',
              fontWeight: 'bold'
            }}>
              {alerts.length}
            </span>
          )}
        </h2>
        
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {alerts.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              color: '#bbb', 
              padding: '40px',
              fontSize: '1.1em'
            }}>
              <CheckCircle size={32} color="#4caf50" style={{ marginBottom: '15px' }} />
              <div>No alerts detected yet</div>
              <div style={{ fontSize: '0.9em', marginTop: '10px' }}>
                Monitoring for Golden Cross and Death Cross patterns...
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {alerts.map((alert, index) => (
                <div key={index} style={{
                  background: 'rgba(20, 20, 20, 0.8)',
                  border: `2px solid ${getPatternColor(alert.pattern)}`,
                  borderRadius: '12px',
                  padding: '20px',
                  position: 'relative'
                }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: '15px'
                  }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px'
                    }}>
                      {getPatternIcon(alert.pattern)}
                      <div>
                        <div style={{ 
                          fontSize: '1.2em', 
                          fontWeight: 'bold', 
                          color: getPatternColor(alert.pattern)
                        }}>
                          {alert.symbol} - {alert.pattern}
                        </div>
                        <div style={{ fontSize: '0.9em', color: '#bbb' }}>
                          {formatTime(alert.timestamp)}
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '1.3em', fontWeight: 'bold', color: '#fff' }}>
                        {formatPrice(alert.price)}
                      </div>
                    </div>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    <div>
                      <div style={{ fontSize: '0.8em', color: '#bbb', marginBottom: '5px' }}>SMA 50</div>
                      <div style={{ fontSize: '1em', fontWeight: 'bold', color: '#fff' }}>
                        {formatPrice(alert.sma_short)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8em', color: '#bbb', marginBottom: '5px' }}>SMA 200</div>
                      <div style={{ fontSize: '1em', fontWeight: 'bold', color: '#fff' }}>
                        {formatPrice(alert.sma_long)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Server Stats */}
      <div style={{
        background: 'rgba(30, 30, 30, 0.95)',
        borderRadius: '15px',
        padding: '25px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <h2 style={{ 
          margin: '0 0 20px 0', 
          color: '#fff',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <Signal size={24} />
          Server Status
        </h2>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '20px' 
        }}>
          <div>
            <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>Monitoring</div>
            <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
              {serverStats.symbols?.join(', ') || 'Loading...'}
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>SMA Windows</div>
            <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
              {serverStats.sma_windows ? 
                `${serverStats.sma_windows.short} / ${serverStats.sma_windows.long}` : 
                'Loading...'
              }
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>Check Interval</div>
            <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
              {serverStats.check_interval ? `${serverStats.check_interval}s` : 'Loading...'}
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>Total Alerts</div>
            <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
              {serverStats.total_alerts || 0}
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>Active Connections</div>
            <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#4caf50' }}>
              {serverStats.active_connections || 0}
            </div>
          </div>
          
          <div>
            <div style={{ fontSize: '0.9em', color: '#bbb', marginBottom: '5px' }}>Last Update</div>
            <div style={{ fontSize: '1.1em', fontWeight: 'bold', color: '#fff' }}>
              {formatTime(lastUpdate)}
            </div>
          </div>
        </div>
      </div>

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          
          .animate-spin {
            animation: spin 1s linear infinite;
          }
          
          /* Custom scrollbar */
          div::-webkit-scrollbar {
            width: 6px;
          }
          
          div::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
          }
          
          div::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
          }
          
          div::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
          }
        `}
      </style>
    </div>
  );
}
