export default function Sidebar({ sessionInfo, symbols, themes, emotions }) {
  return (
    <aside id="sidebar">
      <div className="sidebar-section">
        <h3>Session</h3>
        <div className="session-info">
          <p>Mode <span>{sessionInfo.mode}</span></p>
          <p>Turn <span>{sessionInfo.turn}</span></p>
          <p>Model <span>{sessionInfo.model?.split(':')[1] || sessionInfo.model}</span></p>
        </div>
      </div>

      <div className="sidebar-section">
        <h3>Symbols</h3>
        {symbols.length === 0 ? (
          <p className="empty-state">No symbols yet</p>
        ) : (
          <ul className="symbols-list">
            {symbols.map((s, i) => (
              <li key={i}>
                {s.text || s}
                {s.archetype && <span className="archetype">[{s.archetype}]</span>}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="sidebar-section">
        <h3>Themes</h3>
        {themes.length === 0 ? (
          <p className="empty-state">No themes yet</p>
        ) : (
          <div className="tags">
            {themes.map((t, i) => (
              <span key={i} className="tag">{t}</span>
            ))}
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <h3>Emotions</h3>
        {emotions.length === 0 ? (
          <p className="empty-state">No emotions yet</p>
        ) : (
          <div className="tags">
            {emotions.map((e, i) => (
              <span key={i} className="tag">{e}</span>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}
