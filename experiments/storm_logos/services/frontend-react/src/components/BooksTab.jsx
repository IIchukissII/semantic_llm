import { useState, useEffect } from 'react'
import * as api from '../api'

export default function BooksTab() {
  const [books, setBooks] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [formData, setFormData] = useState({ title: '', author: '', text: '' })
  const [sortBy, setSortBy] = useState('author')
  const [search, setSearch] = useState('')

  useEffect(() => {
    loadBooks()
  }, [])

  async function loadBooks() {
    try {
      const data = await api.getCorpusBooks()
      setBooks(data.books || [])
      if (data.error) setError(data.error)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!formData.text.trim()) return

    setProcessing(true)
    setError(null)
    try {
      const result = await api.processBook(formData.text, formData.title, formData.author)
      if (result.error) {
        setError(result.error)
      } else {
        setFormData({ title: '', author: '', text: '' })
        setShowForm(false)
        loadBooks()
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setProcessing(false)
    }
  }

  if (loading) {
    return <div className="books-tab"><p className="loading-text">Loading corpus...</p></div>
  }

  return (
    <div className="books-tab">
      <div className="books-header">
        <h2>Corpus Library</h2>
        <div className="books-header-right">
          <span className="book-count">{books.length} books</span>
          <button onClick={() => setShowForm(!showForm)}>
            {showForm ? 'Cancel' : '+ Add Book'}
          </button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      <div className="books-controls">
        <input
          type="text"
          placeholder="Search books..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="search-input"
        />
        <select value={sortBy} onChange={e => setSortBy(e.target.value)}>
          <option value="author">Sort by Author</option>
          <option value="title">Sort by Title</option>
          <option value="bonds">Sort by Bonds</option>
          <option value="genre">Sort by Genre</option>
        </select>
      </div>

      {showForm && (
        <form className="add-book-form" onSubmit={handleSubmit}>
          <div className="form-row">
            <input
              type="text"
              placeholder="Title"
              value={formData.title}
              onChange={e => setFormData({...formData, title: e.target.value})}
            />
            <input
              type="text"
              placeholder="Author"
              value={formData.author}
              onChange={e => setFormData({...formData, author: e.target.value})}
            />
          </div>
          <textarea
            placeholder="Paste book text here... (min 100 characters)"
            value={formData.text}
            onChange={e => setFormData({...formData, text: e.target.value})}
            rows={8}
          />
          <button type="submit" disabled={processing || formData.text.length < 100}>
            {processing ? 'Processing...' : `Process (${formData.text.length} chars)`}
          </button>
        </form>
      )}

      <div className="books-list">
        {[...books]
          .filter(book => {
            if (!search) return true
            const q = search.toLowerCase()
            return (book.title || '').toLowerCase().includes(q) ||
                   (book.author || '').toLowerCase().includes(q) ||
                   (book.genre || '').toLowerCase().includes(q)
          })
          .sort((a, b) => {
            if (sortBy === 'bonds') return (b.n_bonds || 0) - (a.n_bonds || 0)
            if (sortBy === 'title') return (a.title || '').localeCompare(b.title || '')
            if (sortBy === 'genre') return (a.genre || '').localeCompare(b.genre || '')
            return (a.author || '').localeCompare(b.author || '')
          })
          .map(book => (
          <div key={book.id} className="book-card">
            <div className="book-author">{book.author}</div>
            <div className="book-title">{book.title}</div>
            <div className="book-meta">
              <span>{book.n_bonds?.toLocaleString() || 0}</span>
              {book.genre && <span className="book-genre">{book.genre}</span>}
            </div>
          </div>
        ))}
      </div>

      {books.length === 0 && !error && !showForm && (
        <div className="empty-state-box">
          <p>No books in corpus yet</p>
          <p className="hint">Click "+ Add Book" to process text or run CLI:</p>
          <code>python -m storm_logos.scripts.process_books --priority</code>
        </div>
      )}
    </div>
  )
}
