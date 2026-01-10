import { useState } from 'react'

export default function AuthModal({ onLogin, onRegister, onSkip }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    try {
      await onLogin(username, password)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleRegister() {
    if (!username || !password) {
      setError('Please fill in all fields')
      return
    }
    setError('')
    try {
      await onRegister(username, password)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="modal">
      <div className="modal-content">
        <h2>Welcome</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
          <div className="auth-buttons">
            <button type="submit">Login</button>
            <button type="button" onClick={handleRegister}>Register</button>
          </div>
          <button type="button" className="secondary" onClick={onSkip}>
            Continue as Guest
          </button>
        </form>
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  )
}
