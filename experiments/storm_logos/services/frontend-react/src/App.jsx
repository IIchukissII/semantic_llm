import { useState, useEffect } from 'react'
import * as api from './api'
import AuthModal from './components/AuthModal'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import HistoryModal from './components/HistoryModal'
import ProfileModal from './components/ProfileModal'
import BooksTab from './components/BooksTab'
import DreamTab from './components/DreamTab'

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [user, setUser] = useState(api.getUser())
  const [sessionId, setSessionId] = useState(null)
  const [sessionInfo, setSessionInfo] = useState({ mode: '-', turn: 0, model: '-' })
  const [messages, setMessages] = useState([])
  const [symbols, setSymbols] = useState([])
  const [themes, setThemes] = useState([])
  const [emotions, setEmotions] = useState([])
  const [loading, setLoading] = useState(false)
  const [showAuth, setShowAuth] = useState(!api.getToken())
  const [showHistory, setShowHistory] = useState(false)
  const [showProfile, setShowProfile] = useState(false)

  useEffect(() => {
    loadInfo()
    if (api.getToken()) {
      handleStartSession()
    }
  }, [])

  async function loadInfo() {
    try {
      const info = await api.getInfo()
      setSessionInfo(prev => ({ ...prev, model: info.model }))
    } catch (e) {
      console.error('Failed to load info:', e)
    }
  }

  async function handleLogin(username, password) {
    await api.login(username, password)
    setUser(api.getUser())
    setShowAuth(false)
    handleStartSession()
  }

  async function handleRegister(username, password) {
    await api.register(username, password)
    setUser(api.getUser())
    setShowAuth(false)
    handleStartSession()
  }

  function handleLogout() {
    api.logout()
    setUser(null)
    setSessionId(null)
    setMessages([])
    setShowAuth(true)
  }

  function handleSkipAuth() {
    setShowAuth(false)
    handleStartSession()
  }

  async function handleStartSession(mode = null) {
    setLoading(true)
    try {
      const data = await api.startSession(mode)
      setSessionId(data.session_id)
      setSessionInfo(prev => ({ ...prev, mode: data.mode, turn: data.turn }))
      addMessage('therapist', data.response)
    } catch (e) {
      addMessage('therapist', `Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  function addMessage(type, content) {
    setMessages(prev => [...prev, { type, content, time: new Date().toLocaleTimeString() }])
  }

  async function handleSendMessage(message) {
    if (!sessionId) {
      await handleStartSession()
    }
    addMessage('user', message)
    setLoading(true)
    try {
      const data = await api.sendMessage(sessionId, message)
      addMessage('therapist', data.response)
      setSessionInfo(prev => ({ ...prev, mode: data.mode, turn: data.turn }))
      setSymbols(data.symbols || [])
      setThemes(data.themes || [])
      setEmotions(data.emotions || [])
    } catch (e) {
      addMessage('therapist', `Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handleEndSession() {
    if (!sessionId) return
    setLoading(true)
    try {
      const data = await api.endSession(sessionId)
      let msg = `Session complete.\n\nTurns: ${data.turns}\nSymbols: ${data.symbols.length}\n\n`
      if (data.archetypes?.length > 0) {
        msg += 'Archetypes manifested:\n'
        data.archetypes.forEach(a => {
          msg += `- ${a.archetype}: ${a.symbols.join(', ')} (felt: ${a.emotions.join(', ')})\n`
        })
      }
      addMessage('therapist', msg)
      setSessionId(null)
      setSessionInfo(prev => ({ ...prev, mode: '-', turn: 0 }))
      setSymbols([])
      setThemes([])
      setEmotions([])
    } catch (e) {
      addMessage('therapist', `Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handlePauseSession() {
    if (!sessionId || !api.getToken()) return
    setLoading(true)
    try {
      const data = await api.pauseSession(sessionId)
      addMessage('therapist', `Session paused after ${data.turns} turns. Resume from History.`)
      setSessionId(null)
      setSessionInfo(prev => ({ ...prev, mode: '-', turn: 0 }))
    } catch (e) {
      addMessage('therapist', `Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handleDeleteSession() {
    if (!sessionId) return
    if (!confirm('Delete this session?')) return
    setLoading(true)
    try {
      await api.deleteSession(sessionId)
      addMessage('therapist', 'Session discarded.')
      setSessionId(null)
      setSessionInfo(prev => ({ ...prev, mode: '-', turn: 0 }))
      setSymbols([])
      setThemes([])
      setEmotions([])
    } catch (e) {
      addMessage('therapist', `Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handleResumeSession(sid) {
    setLoading(true)
    try {
      const data = await api.resumeSession(sid)
      setSessionId(data.session_id)
      addMessage('therapist', data.response)
      setSessionInfo(prev => ({ ...prev, mode: data.mode, turn: data.turn }))
      setSymbols(data.symbols || [])
      setThemes(data.themes || [])
      setEmotions(data.emotions || [])
      setShowHistory(false)
    } catch (e) {
      addMessage('therapist', `Error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div id="app">
      {showAuth && (
        <AuthModal
          onLogin={handleLogin}
          onRegister={handleRegister}
          onSkip={handleSkipAuth}
        />
      )}

      {showHistory && (
        <HistoryModal
          onClose={() => setShowHistory(false)}
          onResume={handleResumeSession}
        />
      )}

      {showProfile && (
        <ProfileModal onClose={() => setShowProfile(false)} />
      )}

      <Header
        user={user}
        onHistory={() => setShowHistory(true)}
        onProfile={() => setShowProfile(true)}
        onLogout={handleLogout}
      />

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          Session
        </button>
        <button
          className={`tab ${activeTab === 'dream' ? 'active' : ''}`}
          onClick={() => setActiveTab('dream')}
        >
          Dream
        </button>
        <button
          className={`tab ${activeTab === 'books' ? 'active' : ''}`}
          onClick={() => setActiveTab('books')}
        >
          Library
        </button>
      </div>

      <main>
        {activeTab === 'chat' && (
          <>
            <Sidebar
              sessionInfo={sessionInfo}
              symbols={symbols}
              themes={themes}
              emotions={emotions}
            />
            <Chat
              messages={messages}
              loading={loading}
              onSend={handleSendMessage}
              onEnd={handleEndSession}
              onPause={handlePauseSession}
              onDelete={handleDeleteSession}
            />
          </>
        )}
        {activeTab === 'dream' && <DreamTab />}
        {activeTab === 'books' && <BooksTab />}
      </main>
    </div>
  )
}
