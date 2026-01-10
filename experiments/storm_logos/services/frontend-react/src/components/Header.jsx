export default function Header({ user, onHistory, onProfile, onLogout }) {
  return (
    <header>
      <h1>Storm-Logos</h1>
      <div className="header-right">
        <span className="user-info">{user?.username || 'Guest'}</span>
        {user && (
          <div className="header-buttons">
            <button className="icon-btn" onClick={onHistory} title="History">
              📋
            </button>
            <button className="icon-btn" onClick={onProfile} title="Profile">
              👤
            </button>
            <button className="icon-btn" onClick={onLogout} title="Logout">
              🚪
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
