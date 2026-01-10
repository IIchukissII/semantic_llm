import { useState, useEffect } from 'react'
import * as api from '../api'

export default function DreamTab() {
  const [dreamText, setDreamText] = useState('')
  const [dreamTitle, setDreamTitle] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [savedDreams, setSavedDreams] = useState([])
  const [showSaved, setShowSaved] = useState(false)
  const [selectedDream, setSelectedDream] = useState(null)

  useEffect(() => {
    if (api.getToken()) {
      loadSavedDreams()
    }
  }, [])

  async function loadSavedDreams() {
    try {
      const result = await api.listDreams()
      setSavedDreams(result.dreams || [])
    } catch (e) {
      console.error('Failed to load dreams:', e)
    }
  }

  async function handleAnalyze() {
    if (!dreamText.trim() || dreamText.length < 20) {
      setError('Dream text too short (min 20 characters)')
      return
    }

    setLoading(true)
    setError(null)
    setAnalysis(null)
    setSelectedDream(null)

    try {
      const result = await api.analyzeDream(dreamText)
      if (result.error) {
        setError(result.error)
      } else {
        setAnalysis(result)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    if (!analysis || !api.getToken()) return

    setSaving(true)
    try {
      const result = await api.saveDream({
        dream: dreamText,
        title: dreamTitle || `Dream ${new Date().toLocaleDateString()}`,
        interpretation: analysis.interpretation,
        symbols: analysis.symbols,
        archetypes: analysis.archetypes,
        dominant_archetype: analysis.dominant_archetype,
      })

      if (result.success) {
        loadSavedDreams()
        setError(null)
      } else {
        setError(result.error || 'Failed to save')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(dreamId) {
    if (!confirm('Delete this dream?')) return

    try {
      await api.deleteDream(dreamId)
      loadSavedDreams()
      if (selectedDream?.id === dreamId) {
        setSelectedDream(null)
      }
    } catch (e) {
      setError(e.message)
    }
  }

  function handleSelectDream(dream) {
    setSelectedDream(dream)
    setShowSaved(false)
  }

  function formatArchetype(name) {
    if (!name) return ''
    return name.replace('_', '/').replace(/\b\w/g, c => c.toUpperCase())
  }

  const isLoggedIn = !!api.getToken()

  return (
    <div className="dream-tab">
      <div className="dream-header">
        <div className="dream-header-left">
          <h2>Dream Analysis</h2>
          <p className="hint">Paste your dream for quick Jungian interpretation</p>
        </div>
        {isLoggedIn && savedDreams.length > 0 && (
          <button
            className="saved-dreams-btn"
            onClick={() => setShowSaved(!showSaved)}
          >
            Saved ({savedDreams.length})
          </button>
        )}
      </div>

      {/* Saved Dreams Panel */}
      {showSaved && (
        <div className="saved-dreams-panel">
          <h3>Saved Dreams</h3>
          <div className="saved-dreams-list">
            {savedDreams.map(dream => (
              <div key={dream.id} className="saved-dream-item">
                <div
                  className="saved-dream-content"
                  onClick={() => handleSelectDream(dream)}
                >
                  <div className="saved-dream-title">{dream.title}</div>
                  <div className="saved-dream-meta">
                    {formatArchetype(dream.dominant_archetype)} | {dream.timestamp?.split('T')[0]}
                  </div>
                  <div className="saved-dream-preview">{dream.text}</div>
                </div>
                <button
                  className="delete-btn"
                  onClick={() => handleDelete(dream.id)}
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Selected saved dream view */}
      {selectedDream && (
        <div className="selected-dream">
          <div className="selected-dream-header">
            <h3>{selectedDream.title}</h3>
            <button onClick={() => setSelectedDream(null)}>Back to new</button>
          </div>
          <div className="analysis-section">
            <h3>Dream</h3>
            <p className="interpretation-text">{selectedDream.text}</p>
          </div>
          <div className="analysis-section">
            <h3>Dominant: {formatArchetype(selectedDream.dominant_archetype)}</h3>
          </div>
          {selectedDream.symbols?.length > 0 && (
            <div className="analysis-section">
              <h3>Symbols</h3>
              <div className="symbols-grid">
                {selectedDream.symbols.map((sym, i) => (
                  <div key={i} className="symbol-card">
                    <div className="symbol-text">{sym.text}</div>
                    {sym.archetype && (
                      <div className="symbol-archetype">{formatArchetype(sym.archetype)}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="analysis-section interpretation">
            <h3>Interpretation</h3>
            <div className="interpretation-text">{selectedDream.interpretation}</div>
          </div>
        </div>
      )}

      {/* New dream input */}
      {!selectedDream && (
        <>
          <div className="dream-input-section">
            {isLoggedIn && (
              <input
                type="text"
                placeholder="Dream title (optional)"
                value={dreamTitle}
                onChange={e => setDreamTitle(e.target.value)}
                className="dream-title-input"
              />
            )}
            <textarea
              placeholder="Describe your dream here..."
              value={dreamText}
              onChange={e => setDreamText(e.target.value)}
              rows={6}
              disabled={loading}
            />
            <div className="dream-actions">
              <button
                onClick={handleAnalyze}
                disabled={loading || dreamText.length < 20}
                className="analyze-btn"
              >
                {loading ? 'Analyzing...' : 'Analyze Dream'}
              </button>
              {analysis && isLoggedIn && (
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="save-btn"
                >
                  {saving ? 'Saving...' : 'Save Dream'}
                </button>
              )}
            </div>
          </div>

          {error && <p className="error-text">{error}</p>}

          {analysis && (
            <div className="dream-analysis">
              {/* Dominant Archetype */}
              <div className="analysis-section dominant-archetype">
                <h3>Dominant Archetype</h3>
                <div className="archetype-badge">
                  <span className="archetype-name">{formatArchetype(analysis.dominant_archetype)}</span>
                  <span className="archetype-score">{(analysis.dominant_score * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Semantic Coordinates */}
              <div className="analysis-section coordinates">
                <h3>Semantic Position</h3>
                <div className="coord-grid">
                  <div className="coord-item">
                    <span className="coord-label">Affirmation</span>
                    <span className={`coord-value ${analysis.coordinates.A >= 0 ? 'positive' : 'negative'}`}>
                      {analysis.coordinates.A >= 0 ? '+' : ''}{analysis.coordinates.A.toFixed(2)}
                    </span>
                  </div>
                  <div className="coord-item">
                    <span className="coord-label">Sacred</span>
                    <span className={`coord-value ${analysis.coordinates.S >= 0 ? 'positive' : 'negative'}`}>
                      {analysis.coordinates.S >= 0 ? '+' : ''}{analysis.coordinates.S.toFixed(2)}
                    </span>
                  </div>
                  <div className="coord-item">
                    <span className="coord-label">Abstraction</span>
                    <span className="coord-value">{analysis.coordinates.tau.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Archetype Scores */}
              <div className="analysis-section archetype-scores">
                <h3>Archetype Presence</h3>
                <div className="archetype-bars">
                  {Object.entries(analysis.archetypes)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, score]) => (
                      <div key={name} className="archetype-bar">
                        <span className="bar-label">{formatArchetype(name)}</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                        <span className="bar-value">{(score * 100).toFixed(0)}%</span>
                      </div>
                    ))
                  }
                </div>
              </div>

              {/* Symbols */}
              <div className="analysis-section symbols">
                <h3>Symbols Detected ({analysis.symbols.length})</h3>
                <div className="symbols-grid">
                  {analysis.symbols.map((sym, i) => (
                    <div key={i} className="symbol-card">
                      <div className="symbol-text">{sym.text}</div>
                      {sym.archetype && (
                        <div className="symbol-archetype">{formatArchetype(sym.archetype)}</div>
                      )}
                      {sym.interpretation && (
                        <div className="symbol-interp">{sym.interpretation}</div>
                      )}
                      <div className="symbol-coords">
                        A: {sym.A.toFixed(2)} | S: {sym.S.toFixed(2)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Interpretation */}
              <div className="analysis-section interpretation">
                <h3>Interpretation</h3>
                <div className="interpretation-text">
                  {analysis.interpretation}
                </div>
              </div>

              {/* Corpus Resonances */}
              {analysis.corpus_resonances && analysis.corpus_resonances.length > 0 && (
                <div className="analysis-section resonances">
                  <h3>Corpus Resonances</h3>
                  <div className="resonance-list">
                    {analysis.corpus_resonances.map((r, i) => (
                      <div key={i} className="resonance-item">
                        <span className="resonance-symbol">"{r.symbol}"</span>
                        <span className="resonance-arrow">-></span>
                        <span className="resonance-source">{r.book}</span>
                        <span className="resonance-author">({r.author})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Markers */}
              {(analysis.markers.transformation > 0 || analysis.markers.journey > 0 || analysis.markers.confrontation > 0) && (
                <div className="analysis-section markers">
                  <h3>Dream Markers</h3>
                  <div className="markers-list">
                    {analysis.markers.transformation > 0 && <span className="marker">Transformation</span>}
                    {analysis.markers.journey > 0 && <span className="marker">Journey</span>}
                    {analysis.markers.confrontation > 0 && <span className="marker">Confrontation</span>}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
