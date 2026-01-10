import { useState, useRef, useEffect } from 'react'

export default function Chat({ messages, loading, onSend, onEnd, onPause, onDelete }) {
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('hybrid')
  const messagesRef = useRef(null)

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    }
  }, [messages])

  function handleSubmit(e) {
    e.preventDefault()
    if (!input.trim()) return
    onSend(input.trim())
    setInput('')
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <section id="chat-area">
      <div id="messages" ref={messagesRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.type}`}>
            <div className="message-content">
              {msg.content.split('\n').map((line, j) => (
                <span key={j}>{line}<br /></span>
              ))}
            </div>
            <div className="message-meta">{msg.time}</div>
          </div>
        ))}
      </div>

      <div id="input-area">
        <textarea
          placeholder="Share a dream, or just start talking..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={3}
        />
        <div className="input-buttons">
          <select value={mode} onChange={e => setMode(e.target.value)}>
            <option value="hybrid">Auto</option>
            <option value="dream">Dream</option>
            <option value="therapy">Therapy</option>
          </select>
          <button onClick={handleSubmit} disabled={loading}>
            {loading ? '...' : 'Send'}
          </button>
          <button className="secondary" onClick={onPause}>Pause</button>
          <button className="secondary" onClick={onEnd}>End</button>
          <button className="secondary" onClick={onDelete}>Delete</button>
        </div>
      </div>
    </section>
  )
}
