import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import ReactMarkdown from 'react-markdown'
import './styles.css'

const initialMessages = [
  { role: 'assistant', time: '09:42', content: '## Conscious Coding Agent\n\n作業フォルダを指定して、自然文で指示してください。日本語とEnglishが混ざった依頼にも対応します。' },
]
const WORKSPACE_HISTORY_KEY = 'conscious-agent-workspaces'
const WORKSPACE_THREADS_KEY = 'conscious-agent-workspace-threads'

const latexSymbols = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', Delta: 'Δ', epsilon: 'ε',
  eta: 'η', theta: 'θ', lambda: 'λ', mu: 'μ', pi: 'π', rho: 'ρ', sigma: 'σ',
  tau: 'τ', phi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω', infty: '∞',
}

function latexToMathML(source, display = false) {
  const normalized = source.replaceAll('¥', '\\').replaceAll('\\cdot', '·').trim()
  let cursor = 0

  const escape = (value) => value.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[char]))
  const atom = (value) => {
    if (/^\d+(?:\.\d+)?$/.test(value)) return `<mn>${escape(value)}</mn>`
    return `<mi>${escape(value)}</mi>`
  }
  const readGroup = () => {
    while (/\s/.test(normalized[cursor] || '')) cursor += 1
    if (normalized[cursor] !== '{') return readAtom()
    cursor += 1
    const value = readSequence('}')
    if (normalized[cursor] === '}') cursor += 1
    return `<mrow>${value}</mrow>`
  }
  const readAtom = () => {
    while (/\s/.test(normalized[cursor] || '')) cursor += 1
    if (normalized[cursor] === '{') return readGroup()
    if (normalized[cursor] === '\\') {
      cursor += 1
      const match = normalized.slice(cursor).match(/^[A-Za-z]+/)
      const command = match ? match[0] : normalized[cursor++]
      if (match) cursor += command.length
      if (command === 'frac') return `<mfrac>${readGroup()}${readGroup()}</mfrac>`
      if (command === 'sqrt') return `<msqrt>${readGroup()}</msqrt>`
      if (command === 'mathbf') return `<mstyle mathvariant="bold">${readGroup()}</mstyle>`
      if (command === 'cdot') return '<mo>·</mo>'
      if (command === 'times') return '<mo>×</mo>'
      if (command === 'leq') return '<mo>≤</mo>'
      if (command === 'geq') return '<mo>≥</mo>'
      if (command === 'neq') return '<mo>≠</mo>'
      return atom(latexSymbols[command] || command)
    }
    const character = normalized[cursor++]
    if (/[A-Za-z]/.test(character)) return atom(character)
    if (/[0-9]/.test(character)) {
      const rest = normalized.slice(cursor).match(/^\d*(?:\.\d+)?/)[0]
      cursor += rest.length
      return atom(character + rest)
    }
    if (character === '·') return '<mo>·</mo>'
    if ('=+-*/(),:'.includes(character)) return `<mo>${escape(character)}</mo>`
    return `<mo>${escape(character)}</mo>`
  }
  const readSequence = (stop) => {
    let output = ''
    while (cursor < normalized.length && normalized[cursor] !== stop) {
      if (/\s/.test(normalized[cursor])) { cursor += 1; continue }
      const base = readAtom()
      let decorated = base
      if (normalized[cursor] === '^' || normalized[cursor] === '_') {
        const operator = normalized[cursor++]
        const exponent = readGroup()
        decorated = operator === '^' ? `<msup>${base}${exponent}</msup>` : `<msub>${base}${exponent}</msub>`
      }
      output += decorated
    }
    return output
  }
  const body = readSequence()
  return `<math class="latex-math" xmlns="http://www.w3.org/1998/Math/MathML" display="${display ? 'block' : 'inline'}"><mrow>${body}</mrow></math>`
}

function MarkdownMessage({ content }) {
  const pattern = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$[^$\n]+\$)/g
  const parts = content.split(pattern).filter((part) => part !== '')
  return <>{parts.map((part, index) => {
    const block = part.startsWith('$$') || part.startsWith('\\[')
    const inline = part.startsWith('\\(') || (part.startsWith('$') && part.endsWith('$'))
    if (block || inline) {
      const formula = block ? part.replace(/^\$\$|\$\$$/g, '').replace(/^\\\[|\\\]$/g, '') : part.replace(/^\\\(|\\\)$/g, '').replace(/^\$|\$$/g, '')
      return <span className={block ? 'latex-block' : 'latex-inline'} key={`${index}-${part}`} dangerouslySetInnerHTML={{ __html: latexToMathML(formula, block) }} />
    }
    return <ReactMarkdown key={`${index}-${part}`}>{part}</ReactMarkdown>
  })}</>
}

// Vite proxies /api to the local Python bridge, avoiding host-name/CORS mismatches.
const API_BASE = window.electronAPI ? 'http://127.0.0.1:8787/api' : '/api'

// The theorem-discovery bundle. Its output is long and is markdown, so it goes
// to the transcript rather than into the panel; what stays here is what it is
// and the handful of things you can ask it to do.
function TheoremPanel({ status, onTheorem }) {
  const [query, setQuery] = useState('assoc')
  const [cycles, setCycles] = useState('12')
  if (!status) return null
  if (!status.available) return <>
    <div className="panel-title spaced">THEOREMS</div>
    <div className="semantic-empty">定理発見システムが見つかりません（{status.path}）。</div>
  </>
  const domains = Object.entries(status.domains || {}).map(([name, count]) => `${name}:${count}`).join(' · ')
  return <>
    <div className="panel-title spaced">THEOREMS</div>
    <div className="context-row"><span>コーパス</span><b title={status.path}>{status.corpus} 件</b></div>
    <div className="template-note">{domains}</div>
    <div className="theorem-row">
      <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="assoc" />
      <button onClick={() => onTheorem('search', { query }, `/theorem search ${query}`)}>検索</button>
    </div>
    <div className="theorem-row">
      <input value={cycles} onChange={(event) => setCycles(event.target.value.replace(/\D/g, ''))} placeholder="12" />
      <button onClick={() => onTheorem('run', { cycles: Number(cycles) || 12 }, `/theorem run ${cycles}`)}>発見核を回す</button>
    </div>
    <div className="answer-row">
      <button className="ghost" onClick={() => onTheorem('obligations', { cycles: Number(cycles) || 12 }, `/theorem obligations ${cycles}`)}>証明義務</button>
      <button className="ghost" onClick={() => onTheorem('verify', {}, '/theorem verify')}>バンドルを検証</button>
    </div>
    <div className="invariant-note">篩を通った本数（asserted）と証明された本数（valid）は別に出ます。Lean が呼ばれていなければ、証明は 0 です。</div>
  </>
}

// Two languages, one box. The dialogue one is calls into the session; the
// physics one declares a field and an equation and draws what comes out.
const EXAMPLE_SCRIPTS = {
  dialogue: 'open_episode("lab")\nask("64x64のケースで処理時間ではなくフガ率をならして")\nno()\nrerun()',
  physics: 'Bz = 1.0\nm = 1.0\nvector: B = (0, 0, Bz)\nlorentz: F exists\ncalc(m d^2 x / dt^2 = lorentz(B))\ndraw(x)',
}

// A conversation written down as calls. Nothing here is evaluated -- the bridge
// parses the source against a fixed table -- and a script with one bad line does
// not run its good ones, so what you read is what happened.
function ScriptEditor({ script, onAction, onSend }) {
  const [source, setSource] = useState('')
  const run = () => {
    const text = source.trim()
    if (text) onAction('script', { source: text }, text)
  }
  return <>
    <div className="panel-title spaced">SCRIPT</div>
    <div className="template-note">対話を呼び出しで、または場と式を宣言して書けます。どちらの言語かは構文解析の結果で決まります。評価はされません。<b>1行でも読めなければ1行も実行しません。</b></div>
    <textarea className="script-source" rows="6" value={source} spellCheck="false"
      placeholder={EXAMPLE_SCRIPTS.dialogue}
      onChange={(event) => setSource(event.target.value)} />
    <div className="answer-row">
      <button onClick={run} disabled={!source.trim()}>実行</button>
      <button className="ghost" onClick={() => setSource(EXAMPLE_SCRIPTS.dialogue)}>対話の例</button>
      <button className="ghost" onClick={() => setSource(EXAMPLE_SCRIPTS.physics)}>物理の例</button>
      <button className="ghost" onClick={() => setSource('')}>消す</button>
    </div>
    {script && <div className={`script-result ${script.ok ? '' : 'failed'}`}>
      {script.needs && script.needs.length > 0 && <div className="script-needs">
        <b>値が決まっていない記号</b>
        <div className="chip-row">{script.needs.map((name) => <span className="chip" key={name}>{name}</span>)}</div>
        <small>推測しないので実行していません。値を書き足してください。</small>
      </div>}
      {script.ok
        ? script.results.map((item) => <div className="script-row" key={item.line}><code>{item.source}</code><span>{item.status || script.language}</span><small>{item.detail}</small></div>)
        : script.error && <div className="script-error">{script.error}</div>}
    </div>}
  </>
}

// A verified form, filled in and then edited. The slots are the parts the
// contract check varies; the textarea is there because a template nobody can
// change is a button, and the point is to start from a shape that is known to
// work and then say something of your own.
function TemplatePicker({ templates, onCompose, onAction, onSend }) {
  const [id, setId] = useState('')
  const [values, setValues] = useState({})
  const [draft, setDraft] = useState('')
  // The same move can be written as something you say or as a call. The slots
  // are the same either way; only the surface differs.
  const [asCode, setAsCode] = useState(false)
  const template = templates.find((item) => item.id === id) || null

  const fill = (item, filled, code) => {
    let text = code ? (item.code || item.form) : item.form
    item.slots.forEach((slot) => {
      text = text.split(`{${slot.name}}`).join(filled[slot.name] ?? slot.default)
    })
    return text
  }

  const choose = (nextId, code = asCode) => {
    const item = templates.find((entry) => entry.id === nextId)
    setId(nextId)
    if (!item) { setValues({}); setDraft(''); return }
    const filled = Object.fromEntries(item.slots.map((slot) => [slot.name, slot.default]))
    setValues(filled)
    setDraft(fill(item, filled, code))
  }

  const editSlot = (name, value) => {
    const filled = { ...values, [name]: value }
    setValues(filled)
    setDraft(fill(template, filled, asCode))
  }

  const toggleCode = (code) => {
    setAsCode(code)
    if (template) setDraft(fill(template, values, code))
  }

  const send = () => {
    const text = draft.trim()
    if (!text || !template) return
    if (asCode) onAction('script', { source: text }, text)
    else if (template.kind === 'REQUEST') onAction('ask', { message: text }, text)
    else if (template.kind === 'ANSWER') onAction('answer', { text }, text)
    else onSend(text)
  }

  const grouped = ['REQUEST', 'ANSWER', 'COMMAND']
  return <>
    <div className="panel-title spaced">VERIFIED TEMPLATES</div>
    <div className="template-note">契約検証が毎回そのまま再生して動作を確かめている形です。編集してから送れます。</div>
    <select className="template-select" value={id} onChange={(event) => choose(event.target.value)}>
      <option value="">テンプレートを選ぶ…</option>
      {grouped.map((kind) => <optgroup label={kind} key={kind}>
        {templates.filter((item) => item.kind === kind).map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
      </optgroup>)}
    </select>
    {template && <div className="template-body">
      <div className="template-meta">{template.note}<em>{template.verified_by}</em></div>
      {template.slots.map((slot) => <label className="template-slot" key={slot.name}>
        <span>{slot.label}</span>
        {slot.options.length > 0
          ? <select value={values[slot.name] ?? slot.default} onChange={(event) => editSlot(slot.name, event.target.value)}>{slot.options.map((option) => <option value={option} key={option}>{option}</option>)}</select>
          : <input value={values[slot.name] ?? slot.default} onChange={(event) => editSlot(slot.name, event.target.value)} />}
      </label>)}
      <div className="template-modes">
        <button className={asCode ? '' : 'selected'} onClick={() => toggleCode(false)}>文として</button>
        <button className={asCode ? 'selected' : ''} onClick={() => toggleCode(true)} disabled={!template.code}>コードとして</button>
      </div>
      <textarea className={`template-draft ${asCode ? 'code' : ''}`} rows="3" value={draft} onChange={(event) => setDraft(event.target.value)} />
      <div className="answer-row">
        <button onClick={send}>送信</button>
        <button className="ghost" onClick={() => onCompose(draft)}>入力欄へ</button>
        <button className="ghost" onClick={() => choose(id)}>スロットから戻す</button>
      </div>
    </div>}
  </>
}

// The interpretation state (L8.27-L8.42) as a panel rather than as prose.
// Every control here is an action the layers already define: answering the
// pending question (L8.33), opening an episode so a blocked question becomes
// askable (L8.42), and naming the model when time and episode both explain the
// evidence (L8.35 leaves it open, L8.36 asks).
function SemanticPanel({ state, templates, theorem, onTheorem, onAction, onCompose, onSend }) {
  if (!state) return <><div className="panel-title">INTERPRETATION</div><div className="semantic-empty">Bridgeに接続すると解釈状態が表示されます。</div></>
  return <>
    <div className="panel-title">INTERPRETATION</div>
    <div className={`semantic-status ${String(state.diagnosis).toLowerCase()}`}><b>{state.diagnosis}</b><small>{state.surface || '対象語なし'}</small></div>
    {state.suspended && <div className="semantic-goal"><span>保留中の目標</span><b>{state.suspended}</b></div>}
    {state.candidates.length > 0 && <div className="chip-row">{state.candidates.map((name) => <span className="chip" key={name}>{name}</span>)}</div>}
    {state.blocked && <div className="semantic-blocked"><b>BLOCKED</b><p>この質問は「{state.episode}」を開かないと答えられません。訊けない質問は高価な質問ではなく、選択肢に入りません。</p><button onClick={() => onAction('open', { episode: state.episode })}>この episode を開く</button></div>}
    {state.question && !state.blocked && <div className="semantic-question"><div className="semantic-question-text">{state.question.text}</div><div className="answer-row"><button onClick={() => onAction('answer', { text: 'はい' })}>はい</button><button onClick={() => onAction('answer', { text: 'いいえ' })}>いいえ</button><button onClick={() => onAction('answer', { text: '分からない' })}>分からない</button><button className="ghost" onClick={() => onCompose('/answer それじゃなくて ')}>訂正…</button></div></div>}
    {state.structural_question && <div className="semantic-question structural"><div className="semantic-question-label">MODEL</div><div className="semantic-question-text">{state.structural_question.questions[0].text}</div><div className="answer-row">{state.structural_question.options.map((model) => <button key={model} onClick={() => onAction('model', { label: model })}>{model}</button>)}</div></div>}
    {templates.length > 0 && <TemplatePicker templates={templates} onCompose={onCompose} onAction={onAction} onSend={onSend} />}
    <ScriptEditor script={state.script} onAction={onAction} onSend={onSend} />
    <TheoremPanel status={theorem} onTheorem={onTheorem} />
    <div className="panel-title spaced">EPISODES</div>
    {state.episodes.map((episode) => <div className={`episode-row ${episode.current ? 'current' : ''}`} key={episode.name}>
      <button className="episode-name" onClick={() => onAction('episode', { episode: episode.name })}><span className={`access-dot ${episode.recallable ? 'open' : 'known'}`} />{episode.name}</button>
      <span className="episode-access">{episode.recallable ? '想起可' : '参照のみ'}</span>
      {!episode.recallable && <button className="episode-open" onClick={() => onAction('open', { episode: episode.name })}>開く</button>}
    </div>)}
    {state.lexicon.length > 0 && <><div className="panel-title spaced">LEXICON</div>{state.lexicon.map((row) => <div className="context-row" key={row.surface}><span>{row.surface}</span><b title={row.observations.join(' / ')}>{row.meaning || row.candidates.join('・') || '-'} <em>{row.state}</em></b></div>)}</>}
    {state.senses.length > 0 && <><div className="panel-title spaced">SENSES</div>{state.senses.map((row) => <div className="context-row" key={row.surface}><span>{row.surface} <em>{row.model}</em></span><b>{Object.entries(row.by_episode).map(([name, meaning]) => `${name}→${meaning || '?'}`).join(' / ')}</b></div>)}</>}
    <div className="panel-title spaced">SESSION INVARIANTS</div>
    <div className="invariant-note">真値を必要としないものだけを数えています。すべて 0 であるべき値です。</div>
    {Object.entries(state.invariants).map(([key, value]) => <div className={`context-row invariant ${value ? 'broken' : ''}`} key={key}><span>{key}</span><b>{value}</b></div>)}
    <div className="panel-title spaced">ACTIONS</div>
    <button className="quick-action" onClick={() => onCompose('/ask ')}><span>✎</span> この要求を解釈にかける</button>
    {state.last_request && state.episodes.length > 1 && <button className="quick-action" onClick={() => onAction('rerun', { request: state.last_request })}><span>⇄</span> 各 episode で実行し直す</button>}
    {state.rerun && <div className="rerun-list">{Object.entries(state.rerun).map(([name, program]) => <div className="rerun-row" key={name}><span>{name}</span><code>{program || '実行せず'}</code></div>)}</div>}
    <button className="quick-action" onClick={() => onCompose('/semantics')}><span>◎</span> 解釈状態をチャットに出力</button>
    <button className="quick-action" onClick={() => onAction('reset', {})}><span>↺</span> 解釈状態をリセット</button>
    <div className="semantic-store">episodes: {state.store}</div>
  </>
}

function App() {
  const [workspace, setWorkspace] = useState('未指定')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState(initialMessages)
  const [workspaces, setWorkspaces] = useState([])
  const [threads, setThreads] = useState([])
  const [activeThreadId, setActiveThreadId] = useState(null)
  const [running, setRunning] = useState(false)
  const [activeTab, setActiveTab] = useState('context')
  const [features, setFeatures] = useState([])
  const [threadMenu, setThreadMenu] = useState(null)
  // The interpretation state (L8.27-L8.42). Held here because the panel shows
  // what the session believes right now, not what the last message said.
  const [semantic, setSemantic] = useState(null)
  // The request and reply forms the contract check replays every run. Offered as
  // editable text rather than as fixed buttons: the shape is what was verified,
  // the wording is the person's.
  const [templates, setTemplates] = useState([])
  // The theorem-discovery bundle is a separate process with its own corpus; the
  // panel shows what it is and hands its output to the transcript.
  const [theorem, setTheorem] = useState(null)

  const newThread = (title = 'New thread', threadMessages = initialMessages) => ({
    id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `thread-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    title,
    messages: threadMessages,
    updatedAt: Date.now(),
  })

  const readThreadStore = () => {
    try {
      const value = JSON.parse(localStorage.getItem(WORKSPACE_THREADS_KEY) || '{}')
      return value && typeof value === 'object' ? value : {}
    } catch { return {} }
  }

  const writeThreadStore = (store) => localStorage.setItem(WORKSPACE_THREADS_KEY, JSON.stringify(store))

  const persistCurrentThread = (path = workspace, currentMessages = messages, currentThreadId = activeThreadId) => {
    if (!path || path === '未指定' || !currentThreadId) return
    const store = readThreadStore()
    const current = Array.isArray(store[path]) ? store[path] : []
    const existing = current.find((thread) => thread.id === currentThreadId)
    const nextThread = { id: currentThreadId, title: existing?.title || 'New thread', messages: currentMessages, updatedAt: Date.now() }
    store[path] = [nextThread, ...current.filter((thread) => thread.id !== currentThreadId)]
    writeThreadStore(store)
  }

  const loadWorkspaceThreads = (path) => {
    const store = readThreadStore()
    let next = Array.isArray(store[path]) ? store[path] : []
    if (!next.length) {
      next = [newThread()]
      store[path] = next
      writeThreadStore(store)
    }
    const selected = next[0]
    setThreads(next)
    setActiveThreadId(selected.id)
    setMessages(selected.messages?.length ? selected.messages : initialMessages)
  }

  useEffect(() => {
    try {
      const savedWorkspaces = JSON.parse(localStorage.getItem(WORKSPACE_HISTORY_KEY) || '[]')
      setWorkspaces(Array.isArray(savedWorkspaces) ? savedWorkspaces : [])
    } catch { setWorkspaces([]) }
    const saved = localStorage.getItem('conscious-agent-session')
    if (!saved) return
    try {
      const session = JSON.parse(saved)
      if (session.workspace) {
        setWorkspace(session.workspace)
        const store = readThreadStore()
        if (!store[session.workspace] && session.messages?.length) {
          const migrated = newThread('Migrated session', session.messages)
          store[session.workspace] = [migrated]
          writeThreadStore(store)
        }
        loadWorkspaceThreads(session.workspace)
      }
    } catch { /* ignore a damaged session and start clean */ }
  }, [])

  const rememberWorkspace = (path) => {
    if (!path || path === '未指定') return
    setWorkspaces((current) => {
      const next = [path, ...current.filter((item) => item !== path)].slice(0, 8)
      localStorage.setItem(WORKSPACE_HISTORY_KEY, JSON.stringify(next))
      return next
    })
  }

  const activateWorkspace = (path, syncBridge = false) => {
    if (!path || path === '未指定') return
    persistCurrentThread()
    setWorkspace(path)
    rememberWorkspace(path)
    loadWorkspaceThreads(path)
    if (syncBridge) fetch(`${API_BASE}/workspace`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }).catch(() => {})
  }

  const selectWorkspace = (path) => {
    activateWorkspace(path, true)
  }

  const removeWorkspace = (path) => {
    setWorkspaces((current) => {
      const next = current.filter((item) => item !== path)
      localStorage.setItem(WORKSPACE_HISTORY_KEY, JSON.stringify(next))
      return next
    })
  }

  useEffect(() => {
    fetch(`${API_BASE}/health`).then((response) => response.json()).then((data) => {
      if (data.workspace) activateWorkspace(data.workspace, false)
    }).catch(() => {})
    fetch(`${API_BASE}/features`).then((response) => response.json()).then((data) => {
      if (Array.isArray(data.features)) setFeatures(data.features)
    }).catch(() => {})
    refreshSemantic()
    fetch(`${API_BASE}/semantic/templates`).then((response) => response.json())
      .then((data) => { if (Array.isArray(data.templates)) setTemplates(data.templates) })
      .catch(() => {})
    fetch(`${API_BASE}/theorem/status`).then((response) => response.json())
      .then((data) => { if (data.status) setTheorem(data.status) }).catch(() => {})
  }, [])

  const refreshSemantic = () => {
    fetch(`${API_BASE}/semantic/state`).then((response) => response.json())
      .then((data) => { if (data.state) setSemantic(data.state) }).catch(() => {})
  }

  const semanticAction = (path, body, echo) => {
    // ``echo`` puts what was sent into the transcript: a template sent from the
    // panel is still something the person said, and a reply with no question
    // above it reads as the agent talking to itself.
    if (echo) {
      const now = new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })
      setMessages((current) => [...current, { role: 'user', time: now, content: echo }])
    }
    setRunning(true)
    fetch(`${API_BASE}/semantic/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
      .then((response) => response.json())
      .then((data) => {
        if (!data.state) return
        setSemantic(data.state)
        if (data.state.reply) {
          const now = new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })
          setMessages((current) => [...current, { role: 'assistant', time: now, content: data.state.reply }])
        }
      })
      .catch(() => {})
      .finally(() => setRunning(false))
  }

  const theoremAction = (path, body, echo) => {
    const now = new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })
    if (echo) setMessages((current) => [...current, { role: 'user', time: now, content: echo }])
    setRunning(true)
    fetch(`${API_BASE}/theorem/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
      .then((response) => response.json())
      .then((data) => {
        if (!data.output) return
        setMessages((current) => [...current, { role: 'assistant', time: now, content: data.output }])
      })
      .catch(() => {})
      .finally(() => setRunning(false))
  }

  const saveSession = () => {
    persistCurrentThread()
    localStorage.setItem('conscious-agent-session', JSON.stringify({ workspace, messages }))
  }

  const createThread = () => {
    persistCurrentThread()
    const thread = newThread()
    const store = readThreadStore()
    const next = [thread, ...(Array.isArray(store[workspace]) ? store[workspace] : threads)]
    store[workspace] = next
    writeThreadStore(store)
    setThreads(next)
    setActiveThreadId(thread.id)
    setMessages(thread.messages)
  }

  const deleteThread = (threadId) => {
    persistCurrentThread()
    const store = readThreadStore()
    const current = Array.isArray(store[workspace]) ? store[workspace] : threads
    let next = current.filter((thread) => thread.id !== threadId)
    if (!next.length) next = [newThread()]
    store[workspace] = next
    writeThreadStore(store)
    setThreads(next)
    setThreadMenu(null)
    if (threadId === activeThreadId) {
      setActiveThreadId(next[0].id)
      setMessages(next[0].messages?.length ? next[0].messages : initialMessages)
    }
  }

  const selectThread = (thread) => {
    persistCurrentThread()
    setThreadMenu(null)
    setActiveThreadId(thread.id)
    setMessages(thread.messages?.length ? thread.messages : initialMessages)
  }

  const chooseWorkspace = async () => {
    try {
      let response
      if (window.electronAPI) {
        const selectedPath = await window.electronAPI.selectWorkspace()
        if (!selectedPath) return
        selectWorkspace(selectedPath)
        return
      } else {
        // Browser/Vite mode uses the Python bridge's native folder picker.
        response = await fetch(`${API_BASE}/workspace/pick`, { method: 'POST' })
      }
      const body = await response.text()
      let data
      try { data = body ? JSON.parse(body) : {} } catch { throw new Error(`bridgeから不正な応答が返りました (${response.status})`) }
      if (!response.ok || data.ok === false) throw new Error(data.error || `workspaceを設定できません (${response.status})`)
      if (!data.workspace) throw new Error('workspace選択の応答が空です。Python bridgeが起動しているか確認してください。')
      selectWorkspace(data.workspace)
    } catch (error) {
      window.alert(`作業フォルダ選択ダイアログを開けませんでした。\n${error.message}`)
    }
  }

  const submit = (event) => {
    event?.preventDefault()
    sendMessage(input.trim())
  }

  const sendMessage = (raw) => {
    const text = (raw || '').trim()
    if (!text || running) return
    const now = new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })
    setMessages((current) => [...current, { role: 'user', time: now, content: text }])
    setInput('')
    setRunning(true)
    fetch(`${API_BASE}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text }) })
      .then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.error || 'agent error'); return data })
      .then((data) => {
        if (data.workspace && data.workspace !== workspace) activateWorkspace(data.workspace, false)
        setMessages((current) => [...current, { role: 'assistant', time: now, content: data.text }])
        if (data.semantic) setSemantic(data.semantic)
        else refreshSemantic()
      })
      .catch((error) => setMessages((current) => [...current, { role: 'assistant', time: now, content: `### 接続エラー\n\n${error.message}\n\nBridgeが起動しているか確認してください。` }]))
      .finally(() => setRunning(false))
  }

  const clearSession = () => {
    const remaining = threads.filter((thread) => thread.id !== activeThreadId)
    const next = remaining.length ? remaining : [newThread()]
    const store = readThreadStore()
    store[workspace] = next
    writeThreadStore(store)
    setThreads(next)
    setActiveThreadId(next[0].id)
    setMessages(next[0].messages)
  }

  useEffect(() => {
    persistCurrentThread()
    if (workspace && workspace !== '未指定' && activeThreadId) {
      setThreads((current) => current.map((thread) => thread.id === activeThreadId ? { ...thread, messages, updatedAt: Date.now() } : thread))
    }
  }, [messages])

  useEffect(() => {
    const closeMenu = () => setThreadMenu(null)
    window.addEventListener('click', closeMenu)
    window.addEventListener('keydown', closeMenu)
    return () => {
      window.removeEventListener('click', closeMenu)
      window.removeEventListener('keydown', closeMenu)
    }
  }, [])

  const contextItems = useMemo(() => [
    ['Workspace', workspace],
    ['Mode', 'Scientific loop'],
    ['Evidence', 'Code + runtime'],
    ['Verification', features.length ? `${features.length} commands` : 'loading'],
    ['Session', `${messages.length} messages`],
  ], [workspace, messages.length, features.length])

  const featureByCommand = (command) => features.find((feature) => feature.command === command)
  const verificationActions = [
    featureByCommand('/self-benchmark') || { command: '/self-benchmark', description: 'L8.6 自己修正ベンチマーク' },
    featureByCommand('/holdout-benchmark') || { command: '/holdout-benchmark', description: 'L8.6.1 hold-out自己修復評価' },
    featureByCommand('/behavior-audit') || { command: '/behavior-audit', description: 'L8.7 行動契約監査' },
  ]

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">C</span><div><strong>Conscious</strong><small>CODING AGENT</small></div></div>
      <button className="workspace-button" onClick={chooseWorkspace}><span>＋</span> 作業フォルダを選択</button>
      <div className="side-label">WORKSPACE</div>
      <div className="workspace-card"><span className="status-dot" /><div><b>{workspace === '未指定' ? 'No workspace' : workspace.split('\\').pop()}</b><small>{workspace}</small></div></div>
      {workspaces.length > 0 && <div className="workspace-history"><div className="side-label">RECENT WORKSPACES</div>{workspaces.map((path) => <div className={`workspace-history-item ${path === workspace ? 'selected' : ''}`} key={path}><button title={`Select ${path}`} onClick={() => selectWorkspace(path)}><span className="status-dot" />{path.split('\\').pop()}</button><button className="remove-workspace" title="Remove workspace" onClick={() => removeWorkspace(path)}>×</button></div>)}</div>}
      {workspace !== '未指定' && <div className="thread-panel"><div className="side-label">THREADS</div><button className="new-thread-button" onClick={createThread}>＋ New thread</button><div className="thread-list">{threads.map((thread) => <button className={`thread-item ${thread.id === activeThreadId ? 'selected' : ''}`} key={thread.id} onClick={() => selectThread(thread)} onContextMenu={(event) => { event.preventDefault(); setThreadMenu({ threadId: thread.id, x: event.clientX, y: event.clientY }) }}><span className="thread-dot" />{thread.title}</button>)}</div></div>}
      {threadMenu && <div className="thread-context-menu" style={{ left: threadMenu.x, top: threadMenu.y }} onClick={(event) => event.stopPropagation()}><button onClick={() => deleteThread(threadMenu.threadId)}>Delete thread</button></div>}
      <div className="side-label">TOOLS</div>
      <nav className="tool-nav">
        <button className="active"><span>◈</span> Agent chat</button>
        <button onClick={() => setInput('/inspect')}><span>⌘</span> Inspect files</button>
        <button onClick={() => setInput('/tests')}><span>✓</span> Run tests</button>
        <button onClick={() => setInput('/self-benchmark')}><span>↻</span> Self repair bench</button>
        <button onClick={() => setInput('/behavior-audit')}><span>◇</span> Behavior audit</button>
        <button onClick={() => setInput('/help')}><span>?</span> Help</button>
      </nav>
      <div className="sidebar-bottom"><button onClick={saveSession}>↓ Save session</button><button onClick={clearSession}>Clear session</button><span>Local-first workspace</span></div>
    </aside>
    <main className="main-panel">
      <header className="topbar"><div><div className="eyebrow">PROJECT WORKSPACE</div><h1>Agent conversation</h1></div><div className="connection"><span className="status-dot" /> Local ready <span className="divider" /> <button onClick={saveSession}>Save</button></div></header>
      <section className="conversation" aria-label="Conversation">
        <div className="conversation-inner">
          {messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.time}-${index}`}>
            <div className="message-meta"><span className="avatar">{message.role === 'assistant' ? 'C' : 'Y'}</span><b>{message.role === 'assistant' ? 'Agent' : 'You'}</b><time>{message.time}</time></div>
            <div className="markdown"><MarkdownMessage content={message.content} /></div>
          </article>)}
          {running && <div className="typing"><span /> <span /> <span /> Agent is thinking</div>}
        </div>
      </section>
      <form className="composer" onSubmit={submit}><div className="composer-label">指示 / INSTRUCTION</div><div className="input-row"><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="作業内容を入力… 例: Explain how the initial state is generated" rows="2" /><button className="send-button" type="submit" disabled={running || !input.trim()}>送信 <span>↗</span></button></div><div className="composer-hint">Enterで送信 · Markdown対応 · 日本語 / English</div></form>
    </main>
    <aside className="inspector"><div className="inspector-tabs"><button className={activeTab === 'context' ? 'selected' : ''} onClick={() => setActiveTab('context')}>Context</button><button className={activeTab === 'activity' ? 'selected' : ''} onClick={() => setActiveTab('activity')}>Activity</button><button className={activeTab === 'semantics' ? 'selected' : ''} onClick={() => setActiveTab('semantics')}>Semantics</button></div>{activeTab === 'semantics' ? <SemanticPanel state={semantic} templates={templates} theorem={theorem} onTheorem={theoremAction} onAction={semanticAction} onCompose={setInput} onSend={sendMessage} /> : activeTab === 'context' ? <><div className="panel-title">SESSION CONTEXT</div><div className="context-list">{contextItems.map(([label, value]) => <div className="context-row" key={label}><span>{label}</span><b title={value}>{value}</b></div>)}</div><div className="panel-title spaced">QUICK ACTIONS</div><button className="quick-action" onClick={() => setInput('このworkspaceの構成を説明して')}><span>⌕</span> Explain workspace</button><button className="quick-action" onClick={() => setInput('Run the tests and summarize the result')}><span>▶</span> Run and summarize</button><div className="panel-title spaced">VERIFICATION</div>{verificationActions.map((feature) => <button className="quick-action" key={feature.command} onClick={() => setInput(feature.command)}><span>◇</span>{feature.description}</button>)}</> : <><div className="panel-title">RECENT ACTIVITY</div><div className="activity-item"><span className="activity-icon">✓</span><div><b>Session initialized</b><small>Ready for instructions</small></div></div><div className="activity-item"><span className="activity-icon">↻</span><div><b>Markdown renderer</b><small>Enabled</small></div></div><div className="activity-item"><span className="activity-icon">◇</span><div><b>Verification tools</b><small>{verificationActions.map((item) => item.command).join(' · ')}</small></div></div></>}</aside>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
