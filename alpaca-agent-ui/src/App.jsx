import { useState } from 'react';
import LiveChart from './livechart.jsx';
import CryptoDashboard from './livechrat_v2.jsx';

function App() {
  const [currentView, setCurrentView] = useState('advanced'); // 'advanced' or 'simple'

  return (
    <div style={{ margin: 0, padding: 0, minHeight: '100vh' }}>
      {/* Dashboard Switcher */}
      <div style={{
        position: 'fixed',
        top: '25px',
        right: '25px',
        zIndex: 1000,
        display: 'flex',
        gap: '10px'
      }}>
        <button
          onClick={() => setCurrentView('advanced')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            border: 'none',
            background: currentView === 'advanced' ? '#4caf50' : '#555',
            color: 'white',
            cursor: 'pointer',
            fontSize: '0.9em',
            fontWeight: 'bold'
          }}
        >
          Advanced Dashboard
        </button>
        <button
          onClick={() => setCurrentView('simple')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            border: 'none',
            background: currentView === 'simple' ? '#4caf50' : '#555',
            color: 'white',
            cursor: 'pointer',
            fontSize: '0.9em',
            fontWeight: 'bold'
          }}
        >
          Simple Dashboard
        </button>
      </div>

      {/* Render appropriate dashboard */}
      {currentView === 'advanced' ? <LiveChart /> : <CryptoDashboard />}
    </div>
  )
}

export default App

