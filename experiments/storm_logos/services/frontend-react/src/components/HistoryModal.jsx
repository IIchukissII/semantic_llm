import { useState, useEffect } from 'react'
import * as api from '../api'

export default function HistoryModal({ onClose, onResume }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadHistory()
  }, [])

  async function loadHistory() {
    try {
      const data = await api.getHistory()
      setSessions(data.sessions || [])
    } catch (e) {
      console.error('Failed to load history:', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal" onClick={e => e.target.className === 'modal' && onClose()}>
      <div className="modal-content large">
        <button className="close-btn" onClick={onClose}>&times;</button>
        <h2>Session History</h2>

        {loading ? (
          <p>Loading...</p>
        ) : sessions.length === 0 ? (
          <p>No previous sessions found.</p>
        ) : (
          <div className="history-list">
            {sessions.map(s => (
              <div key={s.session_id} className={`history-item ${s.status || 'ended'}`}>
                <div className="history-header">
                  <span className="date">
                    {s.timestamp?.substring(0, 16).replace('T', ' ') || 'Unknown'}
                  </span>
                  <span className="status">{s.status || 'ended'}</span>
                </div>
                <div className="history-details">
                  <span className="mode">{s.mode}</span>
                  <span className="summary">{s.summary}</span>
                </div>
                <div className="history-archetypes">
                  Archetypes: {s.archetypes?.join(', ') || 'none'}
                </div>
                {s.status === 'paused' && (
                  <button
                    className="resume-btn"
                    onClick={() => onResume(s.session_id)}
                  >
                    Resume
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
