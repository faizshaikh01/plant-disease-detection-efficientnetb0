import { useRef, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const inputRef = useRef(null)
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [results, setResults] = useState([])
  const [history, setHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const chooseFiles = (selectedFiles) => {
    const selected = Array.from(selectedFiles || []).filter((item) => item.type.startsWith('image/')).slice(0, 20)
    if (!selected.length) {
      setError('Please choose one or more JPG, JPEG, or PNG images.')
      return
    }
    setFiles(selected)
    setPreviews(selected.map((item) => URL.createObjectURL(item)))
    setResults([])
    setError('')
  }

  const predict = async () => {
    if (!files.length) return
    setLoading(true)
    setError('')
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    try {
      const response = await fetch(`${API_URL}/predict-batch`, { method: 'POST', body: formData })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed')
      setResults(data.results)
      loadHistory()
    } catch (err) {
      setError(`Could not connect to the backend. ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const loadHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/history?limit=20`)
      const data = await response.json()
      setHistory(data.items || [])
      setShowHistory(true)
    } catch {
      setError('Could not load prediction history.')
    }
  }

  const clearHistory = async () => {
    if (!window.confirm('Delete all saved prediction history?')) return
    await fetch(`${API_URL}/history`, { method: 'DELETE' })
    setHistory([])
  }

  return (
    <main className="app-shell">
      <nav className="navbar"><div className="brand"><span className="brand-mark">✦</span> LeafLens</div><div className="nav-actions"><button className="history-button" onClick={loadHistory}>History</button><span className="status"><span className="status-dot" /> AI model online</span></div></nav>
      <section className="hero-section">
        <div className="hero-copy"><p className="eyebrow">PLANT HEALTH • COMPUTER VISION</p><h1>Understand what your leaves are telling you.</h1><p className="subtitle">Upload up to 20 leaf images and our EfficientNetB0 model will analyze them together.</p></div>
        <div className="workspace-card">
          <div className={`drop-zone ${files.length ? 'has-file' : ''}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFiles(event.dataTransfer.files) }} onClick={() => inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept="image/*" multiple onChange={(event) => chooseFiles(event.target.files)} hidden />
            {previews.length ? <div className="preview-grid">{previews.slice(0, 6).map((src, index) => <img key={src} src={src} alt={`Leaf ${index + 1}`} />)}</div> : <div className="upload-icon">↑</div>}
            <strong>{files.length ? `${files.length} image${files.length > 1 ? 's' : ''} selected` : 'Drop leaf images here'}</strong>
            <span>or click to browse from your device</span><small>JPG, JPEG or PNG • Up to 20 images</small>
          </div>
          <button className="predict-button" disabled={!files.length || loading} onClick={predict}>{loading ? 'Analyzing images…' : `Analyze ${files.length || ''} image${files.length === 1 ? '' : 's'}`} {!loading && <span>→</span>}</button>
          {error && <p className="error-message">{error}</p>}
        </div>
      </section>
      {results.length > 0 && <section className="results-grid">{results.map((result) => <article className="result-card" key={result.filename}><p className="eyebrow">ANALYSIS RESULT</p>{result.gradcam && <img className="gradcam-image" src={`data:image/jpeg;base64,${result.gradcam}`} alt="Grad-CAM explanation" />}<h2>{result.prediction.class_name.replaceAll('_', ' ')}</h2><p className="filename">{result.filename}</p><div className="confidence-row"><div className="confidence-track"><div style={{ width: `${result.prediction.confidence}%` }} /></div><strong>{result.prediction.confidence}%</strong></div><div className="alternatives">{result.top_predictions.map((item) => <div className="prediction-row" key={item.class_name}><span>{item.class_name.replaceAll('_', ' ')}</span><strong>{item.confidence}%</strong></div>)}</div><p className="explanation-label">Highlighted area: model attention</p></article>)}</section>}
      {showHistory && <section className="history-panel"><div className="history-heading"><div><p className="eyebrow">SAVED ANALYSES</p><h2>Prediction history</h2></div><div className="history-actions"><a className="history-button" href={`${API_URL}/history/export`}>Export CSV</a><button className="history-button danger" onClick={clearHistory}>Clear history</button><button className="close-history" onClick={() => setShowHistory(false)}>Close</button></div></div>{history.length ? history.map((item) => <div className="history-row" key={item.id}><span className="history-file">{item.filename}</span><span>{item.class_name.replaceAll('_', ' ')}</span><strong>{item.confidence}%</strong><time>{new Date(item.created_at).toLocaleString()}</time></div>) : <p className="empty-history">No predictions saved yet.</p>}</section>}
      <footer>Built with React, FastAPI and EfficientNetB0</footer>
    </main>
  )
}

export default App
