import { useState, useEffect } from 'react'
import * as api from '../api'

const ARCHETYPES = [
  'shadow', 'anima_animus', 'self', 'mother',
  'father', 'hero', 'trickster', 'death_rebirth'
]

export default function ProfileModal({ onClose }) {
  const [profile, setProfile] = useState(null)
  const [evolution, setEvolution] = useState([])
  const [selectedArch, setSelectedArch] = useState('shadow')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (selectedArch) {
      loadEvolution(selectedArch)
    }
  }, [selectedArch])

  async function loadProfile() {
    try {
      const data = await api.getProfile()
      setProfile(data)
    } catch (e) {
      console.error('Failed to load profile:', e)
    } finally {
      setLoading(false)
    }
  }

  async function loadEvolution(arch) {
    try {
      const data = await api.getEvolution(arch)
      setEvolution(data)
    } catch (e) {
      console.error('Failed to load evolution:', e)
    }
  }

  if (loading) {
    return (
      <div className="modal" onClick={e => e.target.className === 'modal' && onClose()}>
        <div className="modal-content large">
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  const maxCount = Math.max(...Object.values(profile?.archetypes || {}), 1)

  return (
    <div className="modal" onClick={e => e.target.className === 'modal' && onClose()}>
      <div className="modal-content large">
        <button className="close-btn" onClick={onClose}>&times;</button>
        <h2>Your Archetype Profile</h2>

        <div className="profile-stats">
          <div className="stat-box">
            <div className="stat-value">{profile?.total_sessions || 0}</div>
            <div className="stat-label">Sessions</div>
          </div>
          <div className="stat-box">
            <div className="stat-value">{profile?.dominant_archetypes?.length || 0}</div>
            <div className="stat-label">Dominant Archetypes</div>
          </div>
        </div>

        <div className="archetype-chart">
          {Object.entries(profile?.archetypes || {}).map(([arch, count]) => (
            <div key={arch} className="archetype-bar">
              <span className="name">{arch}</span>
              <div className="bar">
                <div
                  className="fill"
                  style={{ width: `${(count / maxCount) * 100}%` }}
                />
              </div>
              <span className="count">{count}</span>
            </div>
          ))}
        </div>

        <div className="evolution-section">
          <h3>Archetype Evolution</h3>
          <select value={selectedArch} onChange={e => setSelectedArch(e.target.value)}>
            {ARCHETYPES.map(a => (
              <option key={a} value={a}>{a.replace('_', '/')}</option>
            ))}
          </select>

          <div className="evolution-list">
            {evolution.length === 0 ? (
              <p>No manifestations yet.</p>
            ) : (
              evolution.slice(-10).map((item, i) => (
                <div key={i} className="evolution-item">
                  <div className="date">
                    {item.timestamp?.substring(0, 10) || 'Unknown'}
                  </div>
                  <div className="symbols">
                    Symbols: {item.symbols?.join(', ') || 'none'}
                  </div>
                  <div className="emotions">
                    Felt: {item.emotions?.join(', ') || 'none'}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
