/* ── State ────────────────────────────────────────────────────────────────── */
let isStreaming = false;

/* ── Boot ─────────────────────────────────────────────────────────────────── */
(async function init() {
  setupDrop();
  setupFileInput();

  // Restore toggle states from localStorage
  for (const id of ['toggle-mq', 'toggle-hyde', 'toggle-rerank']) {
    const key = id === 'toggle-mq' ? 'multi_query' : id === 'toggle-hyde' ? 'hyde' : 'reranking';
    const saved = localStorage.getItem('rag_' + key);
    if (saved !== null) document.getElementById(id).checked = saved === 'true';
  }

  // If server already has a document loaded, go straight to chat
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ready) {
      applySettings(data.settings);
      showChat(data.doc_name, data.n_parents, data.n_children);
    }
  } catch (_) {}
})();

/* ── Screen transitions ───────────────────────────────────────────────────── */
function showChat(docName, nParents, nChildren) {
  document.getElementById('screen-upload').classList.add('hidden');
  document.getElementById('screen-chat').classList.remove('hidden');

  document.getElementById('sidebar-doc-name').textContent = docName || '—';
  document.getElementById('sidebar-doc-stats').textContent =
    nParents ? `${nParents} passages · ${nChildren} chunks` : '—';
}

function showUpload() {
  document.getElementById('screen-chat').classList.add('hidden');
  document.getElementById('screen-upload').classList.remove('hidden');
  resetProgress();
}

/* ── Drag-and-drop ────────────────────────────────────────────────────────── */
function setupDrop() {
  const zone = document.getElementById('drop-zone');

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  zone.addEventListener('click', e => {
    if (e.target.classList.contains('link-btn')) return;
    document.getElementById('file-input').click();
  });
}

function setupFileInput() {
  document.getElementById('file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
    e.target.value = '';
  });
}

/* ── Upload / ingest ──────────────────────────────────────────────────────── */
async function uploadFile(file) {
  showProgress('Uploading…', 10);
  hideError();

  const fd = new FormData();
  fd.append('file', file);

  try {
    setProgress(30, 'Parsing document…');
    const res = await fetch('/api/ingest/file', { method: 'POST', body: fd });
    setProgress(70, 'Building index…');

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }

    setProgress(100, 'Ready!');
    const data = await res.json();

    await sleep(400);
    showChat(data.doc_name, data.n_parents, data.n_children);
  } catch (e) {
    showError(e.message);
    resetProgress();
  }
}

async function loadSaved() {
  showProgress('Loading saved index…', 20);
  hideError();

  try {
    const res = await fetch('/api/ingest/load', { method: 'POST' });
    setProgress(80, 'Restoring index…');

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'No saved index found');
    }

    setProgress(100, 'Ready!');
    const data = await res.json();
    await sleep(400);
    showChat(data.doc_name, data.n_parents, data.n_children);
  } catch (e) {
    showError(e.message);
    resetProgress();
  }
}

/* ── Progress helpers ─────────────────────────────────────────────────────── */
function showProgress(label, pct) {
  document.getElementById('upload-progress').classList.remove('hidden');
  setProgress(pct, label);
}
function setProgress(pct, label) {
  document.getElementById('progress-bar').style.width = pct + '%';
  if (label) document.getElementById('progress-label').textContent = label;
}
function resetProgress() {
  document.getElementById('upload-progress').classList.add('hidden');
  document.getElementById('progress-bar').style.width = '0%';
}
function showError(msg) {
  const el = document.getElementById('upload-error');
  el.textContent = msg;
  el.classList.remove('hidden');
}
function hideError() {
  document.getElementById('upload-error').classList.add('hidden');
}

/* ── Settings toggles ─────────────────────────────────────────────────────── */
function applySettings(settings) {
  if (settings.multi_query !== undefined)
    document.getElementById('toggle-mq').checked = settings.multi_query;
  if (settings.hyde !== undefined)
    document.getElementById('toggle-hyde').checked = settings.hyde;
  if (settings.reranking !== undefined)
    document.getElementById('toggle-rerank').checked = settings.reranking;
}

async function saveSetting(key, value) {
  localStorage.setItem('rag_' + key, value);
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
  } catch (_) {}
}

/* ── New session ──────────────────────────────────────────────────────────── */
function newSession() {
  clearMessages();
  showUpload();
}

/* ── Input handling ───────────────────────────────────────────────────────── */
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuestion();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

/* ── Query / SSE ──────────────────────────────────────────────────────────── */
async function submitQuestion() {
  if (isStreaming) return;

  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  isStreaming = true;
  document.getElementById('send-btn').disabled = true;

  hideEmptyState();
  appendUserMessage(question);

  const thinkingEl = appendThinking();

  const settings = {
    multi_query: document.getElementById('toggle-mq').checked,
    hyde:        document.getElementById('toggle-hyde').checked,
    reranking:   document.getElementById('toggle-rerank').checked,
  };

  let assistantBubble = null;
  let assistantWrapper = null;
  let streamText = '';
  let pendingSources = null;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, settings }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Query failed');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch (_) { continue; }

        if (event.phase === 'retrieved') {
          thinkingEl.remove();
          if (event.sources?.length) pendingSources = event.sources;
          const result = appendAssistantBubble();
          assistantWrapper = result.wrapper;
          assistantBubble = result.bubble;

        } else if (event.phase === 'token') {
          if (!assistantBubble) {
            thinkingEl.remove();
            const result = appendAssistantBubble();
            assistantWrapper = result.wrapper;
            assistantBubble = result.bubble;
          }
          streamText += event.token;
          renderStreamText(assistantBubble, streamText);

        } else if (event.phase === 'done') {
          if (assistantBubble) {
            renderFinalText(assistantBubble, event.answer || streamText);
          }
          if (pendingSources?.length && assistantWrapper) {
            attachSourcesButton(assistantWrapper, pendingSources);
          }
          const conf = event.confidence;
          if (conf != null && !isNaN(conf) && isFinite(conf)) {
            renderConfidence(conf);
          }

        } else if (event.phase === 'error') {
          thinkingEl.remove();
          appendErrorMessage(event.error || 'Something went wrong.');
        }
      }
    }
  } catch (e) {
    try { thinkingEl.remove(); } catch (_) {}
    appendErrorMessage(e.message);
  } finally {
    isStreaming = false;
    document.getElementById('send-btn').disabled = false;
  }

  scrollToBottom();
}

/* ── Message rendering ────────────────────────────────────────────────────── */
function clearMessages() {
  const msgs = document.getElementById('messages');
  msgs.innerHTML = `
    <div class="empty-state" id="empty-state">
      <div class="empty-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <p class="empty-title" id="empty-title">Ready to answer your questions</p>
      <p class="empty-sub" id="empty-sub">Ask anything about your document below</p>
    </div>`;
}

function hideEmptyState() {
  const es = document.getElementById('empty-state');
  if (es) es.remove();
}

function appendUserMessage(text) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
  msgs.appendChild(div);
  scrollToBottom();
}

function appendThinking() {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'thinking-row';
  div.innerHTML = `<div class="spinner"></div><span class="thinking-text">Searching your document…</span>`;
  msgs.appendChild(div);
  scrollToBottom();
  return div;
}

function appendAssistantBubble() {
  const msgs = document.getElementById('messages');
  const wrapper = document.createElement('div');
  wrapper.className = 'msg msg-assistant';
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = '<span class="cursor"></span>';
  wrapper.appendChild(bubble);
  msgs.appendChild(wrapper);
  scrollToBottom();
  return { wrapper, bubble };
}

function attachSourcesButton(wrapper, sources) {
  const btn = document.createElement('button');
  btn.className = 'sources-toggle-btn';
  btn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
    Sources <span class="sources-count">${sources.length}</span>`;

  const grid = document.createElement('div');
  grid.className = 'sources-grid hidden';

  for (const s of sources) {
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-card-header">
        <span class="source-idx">${s.idx}</span>
        ${s.score != null ? `<span class="source-score">${s.score}</span>` : ''}
      </div>
      <p class="source-fname">${escHtml(s.file_name)}</p>
      <p class="source-excerpt">${escHtml(s.excerpt)}</p>`;
    grid.appendChild(card);
  }

  btn.addEventListener('click', () => {
    const open = grid.classList.toggle('hidden');
    btn.classList.toggle('open', !open);
  });

  wrapper.appendChild(btn);
  wrapper.appendChild(grid);
  scrollToBottom();
}

function renderStreamText(bubble, text) {
  bubble.innerHTML = escHtml(text).replace(/\n/g, '<br>') + '<span class="cursor"></span>';
  scrollToBottom();
}

function renderFinalText(bubble, text) {
  bubble.innerHTML = formatMarkdown(text);
}

function appendErrorMessage(msg) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg msg-assistant';
  div.innerHTML = `<div class="msg-bubble" style="color:var(--red);border-color:rgba(239,68,68,.3)">
    ${escHtml(msg)}
  </div>`;
  msgs.appendChild(div);
  scrollToBottom();
}


function renderConfidence(conf) {
  const msgs = document.getElementById('messages');
  const pct = Math.round(conf * 100);
  const row = document.createElement('div');
  row.className = 'confidence-row';
  row.innerHTML = `
    <span class="conf-label">Confidence</span>
    <div class="conf-bar-track"><div class="conf-bar-fill" style="width:${pct}%"></div></div>
    <span class="conf-pct">${pct}%</span>`;
  msgs.appendChild(row);
  scrollToBottom();
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */
function scrollToBottom() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatMarkdown(text) {
  // Basic markdown: bold, italic, code, code blocks, line breaks
  let s = escHtml(text);
  s = s.replace(/```[\s\S]*?```/g, m => {
    const code = m.slice(3, -3).replace(/^\w*\n/, '');
    return `<pre><code>${code}</code></pre>`;
  });
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  s = s.replace(/\n\n+/g, '</p><p>');
  s = s.replace(/\n/g, '<br>');
  return `<p>${s}</p>`;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }


/* ── Embedding Map ─────────────────────────────────────────────────────────── */

const EMB_COLORS = [
  '#4e9af1','#f15a4e','#4ef19a','#f1c94e',
  '#b34ef1','#f1884e','#4ef1e8','#f14ea9',
];

let _embData  = null;
let _embDim   = '3d';
let _embLevel = 'chunks';   // 'chunks' | 'passages'
let _embThree = null;

function _activePoints() {
  if (!_embData) return [];
  return _embLevel === 'passages' ? (_embData.passages || []) : (_embData.chunks || []);
}

async function openEmbeddingMap() {
  const modal   = document.getElementById('emb-modal');
  const loading = document.getElementById('emb-loading');
  const footer  = document.getElementById('emb-footer');

  modal.classList.remove('hidden');
  loading.classList.remove('hidden');
  loading.textContent = 'Loading embeddings…';
  footer.innerHTML = '';
  document.getElementById('emb-canvas-2d').classList.add('hidden');
  document.getElementById('emb-canvas-3d').classList.remove('hidden');

  // Clear cached data so fresh ingest is always reflected
  _embData = null;

  try {
    const res = await fetch('/api/embeddings');
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to load embeddings');
    }
    _embData = await res.json();
  } catch (e) {
    loading.textContent = 'Error: ' + e.message;
    return;
  }

  if (!(_embData.chunks || []).length) {
    loading.textContent = 'No embeddings found — ingest a document first.';
    return;
  }

  loading.classList.add('hidden');

  // Reset controls to defaults on open
  _embDim   = '3d';
  _embLevel = 'chunks';
  document.getElementById('btn-3d').classList.add('active');
  document.getElementById('btn-2d').classList.remove('active');
  document.getElementById('btn-level-chunks').classList.add('active');
  document.getElementById('btn-level-passages').classList.remove('active');

  _renderEmbLegend();
  _showEmbDim('3d');
}

function closeEmbeddingMap() {
  document.getElementById('emb-modal').classList.add('hidden');
  _destroyThree();
}

function closeEmbeddingMapOutside(e) {
  if (e.target === document.getElementById('emb-modal')) closeEmbeddingMap();
}

function switchEmbDim(dim) {
  if (dim === _embDim) return;
  _embDim = dim;
  document.getElementById('btn-3d').classList.toggle('active', dim === '3d');
  document.getElementById('btn-2d').classList.toggle('active', dim === '2d');
  _showEmbDim(dim);
}

function switchEmbLevel(level) {
  if (level === _embLevel) return;
  _embLevel = level;
  document.getElementById('btn-level-chunks').classList.toggle('active',   level === 'chunks');
  document.getElementById('btn-level-passages').classList.toggle('active', level === 'passages');
  _renderEmbLegend();
  _showEmbDim(_embDim);
}

function _showEmbDim(dim) {
  const pts = _activePoints();
  if (!pts.length) return;
  const c2d = document.getElementById('emb-canvas-2d');
  const c3d = document.getElementById('emb-canvas-3d');
  if (dim === '3d') {
    c2d.classList.add('hidden');
    c3d.classList.remove('hidden');
    _init3D(pts);
  } else {
    _destroyThree();
    c3d.classList.add('hidden');
    c2d.classList.remove('hidden');
    _init2D(pts);
  }
}

function _renderEmbLegend() {
  const data   = _embData;
  const footer = document.getElementById('emb-footer');
  const isPassages = _embLevel === 'passages';
  const n = Math.min(data.n_groups, EMB_COLORS.length);
  let html = '';
  for (let i = 0; i < n; i++) {
    const label = isPassages ? `Passage ${i + 1}` : `Passage ${i + 1} chunks`;
    html += `<span class="emb-legend-item">
      <span class="emb-legend-dot" style="background:${EMB_COLORS[i]}"></span>${label}
    </span>`;
  }
  if (data.n_groups > EMB_COLORS.length) {
    html += `<span class="emb-legend-item" style="color:var(--text-3)">+${data.n_groups - EMB_COLORS.length} more</span>`;
  }
  const pts = _activePoints();
  html += `<span class="emb-variance">${pts.length} ${isPassages ? 'passages' : 'chunks'}`;
  if (data.variance && data.variance.length) {
    const pcts = data.variance.slice(0, 3).map(v => (v * 100).toFixed(1) + '%').join(' + ');
    html += ` · PCA: ${pcts}`;
  }
  html += '</span>';
  footer.innerHTML = html;
}

function _destroyThree() {
  if (_embThree) {
    cancelAnimationFrame(_embThree.rafId);
    _embThree.renderer.dispose();
    _embThree.renderer.forceContextLoss();
    document.getElementById('emb-canvas-3d').innerHTML = '';
    _embThree = null;
  }
}

/* ── 3D — Three.js orbit viewer ─────────────────────────────────────────── */
async function _init3D(pts) {
  _destroyThree();
  const container = document.getElementById('emb-canvas-3d');
  container.innerHTML = '';

  if (!window.THREE) {
    await _loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js');
  }

  const THREE = window.THREE;
  const w = container.clientWidth  || 800;
  const h = container.clientHeight || 500;

  const scene    = new THREE.Scene();
  scene.background = new THREE.Color(0x141414);

  const camera   = new THREE.PerspectiveCamera(55, w / h, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  // Build point cloud geometry — point size larger in passages view (fewer points)
  const positions = new Float32Array(pts.length * 3);
  const colors    = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    positions[i * 3]     = pts[i].x;
    positions[i * 3 + 1] = pts[i].y;
    positions[i * 3 + 2] = pts[i].z;
    const col = new THREE.Color(EMB_COLORS[pts[i].group % EMB_COLORS.length]);
    colors[i * 3]     = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  const pointSize = _embLevel === 'passages' ? 0.09 : 0.055;
  const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true });
  const cloud = new THREE.Points(geo, mat);
  scene.add(cloud);

  // Orbit state
  let theta = 0.4, phi = 1.1, r = 3.2;
  let dragging = false, lastX = 0, lastY = 0, autoRotate = true;

  renderer.domElement.addEventListener('mousedown', e => {
    dragging = true; autoRotate = false;
    lastX = e.clientX; lastY = e.clientY;
  });
  const onMove = e => {
    if (!dragging) return;
    theta -= (e.clientX - lastX) * 0.007;
    phi = Math.max(0.12, Math.min(Math.PI - 0.12, phi + (e.clientY - lastY) * 0.007));
    lastX = e.clientX; lastY = e.clientY;
  };
  const onUp = () => { dragging = false; };
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup',   onUp);
  renderer.domElement.addEventListener('wheel', e => {
    r = Math.max(1.2, Math.min(9, r + e.deltaY * 0.005));
  }, { passive: true });

  // Hover / raycaster
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.06;
  const mouse   = new THREE.Vector2();
  const tooltip = document.getElementById('emb-tooltip');

  renderer.domElement.addEventListener('mousemove', e => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(cloud);
    if (hits.length) {
      const p = pts[hits[0].index];
      tooltip.textContent = p.text;
      tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
      tooltip.style.top  = (e.clientY - rect.top  - 12) + 'px';
      tooltip.classList.remove('hidden');
    } else {
      tooltip.classList.add('hidden');
    }
  });
  renderer.domElement.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));

  // Resize observer
  const ro = new ResizeObserver(() => {
    const nw = container.clientWidth, nh = container.clientHeight;
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  });
  ro.observe(container);

  // Render loop
  function animate() {
    const id = requestAnimationFrame(animate);
    if (_embThree) _embThree.rafId = id;
    if (autoRotate) theta += 0.0035;
    camera.position.set(
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.cos(theta)
    );
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }

  _embThree = { renderer, rafId: null, ro, onMove, onUp };
  animate();
}

/* ── 2D — Canvas scatter with pan + zoom ────────────────────────────────── */
function _init2D(pts) {
  const canvas = document.getElementById('emb-canvas-2d');
  const wrap   = document.getElementById('emb-canvas-wrap');
  canvas.width  = wrap.clientWidth  || 800;
  canvas.height = wrap.clientHeight || 500;

  const ctx = canvas.getContext('2d');
  const PAD = 36;

  let scale = 1, offX = 0, offY = 0;
  let drag = false, dragX = 0, dragY = 0;

  function toScreen(px, py) {
    const usable_w = (canvas.width  - PAD * 2) * scale;
    const usable_h = (canvas.height - PAD * 2) * scale;
    return [
      offX + (px + 1) * 0.5 * usable_w + PAD,
      offY + (1 - (py + 1) * 0.5) * usable_h + PAD,
    ];
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle grid
    ctx.strokeStyle = 'rgba(255,255,255,.04)';
    ctx.lineWidth = 1;
    for (let g = -1; g <= 1.01; g += 0.5) {
      let [x0, y0] = toScreen(g, -1.1), [x1, y1] = toScreen(g, 1.1);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      [x0, y0] = toScreen(-1.1, g); [x1, y1] = toScreen(1.1, g);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }

    // Points — larger dots in passages view
    const baseR = _embLevel === 'passages' ? 7 : 4.5;
    const R = Math.max(3, Math.min(12, baseR * scale));
    for (const p of pts) {
      const [sx, sy] = toScreen(p.x, p.y);
      ctx.beginPath();
      ctx.arc(sx, sy, R, 0, Math.PI * 2);
      ctx.fillStyle   = EMB_COLORS[p.group % EMB_COLORS.length];
      ctx.globalAlpha = 0.88;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  draw();

  // Pan
  const onDown = e => { drag = true; dragX = e.clientX; dragY = e.clientY; };
  const onMove = e => {
    if (!drag) return;
    offX += e.clientX - dragX; offY += e.clientY - dragY;
    dragX = e.clientX; dragY = e.clientY;
    draw();
  };
  const onUp = () => { drag = false; };
  canvas.addEventListener('mousedown', onDown);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup',   onUp);

  // Zoom centred on cursor
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect   = canvas.getBoundingClientRect();
    const mx     = e.clientX - rect.left;
    const my     = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.13 : 1 / 1.13;
    offX = mx - (mx - offX) * factor;
    offY = my - (my - offY) * factor;
    scale = Math.max(0.25, Math.min(10, scale * factor));
    draw();
  }, { passive: false });

  // Hover tooltip
  const tooltip = document.getElementById('emb-tooltip');
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const my   = e.clientY - rect.top;
    const threshold = Math.max(7, 8 * scale);
    let found = null;
    for (const p of pts) {
      const [sx, sy] = toScreen(p.x, p.y);
      if ((mx - sx) ** 2 + (my - sy) ** 2 < threshold * threshold) { found = p; break; }
    }
    if (found) {
      tooltip.textContent = found.text;
      tooltip.style.left = (mx + 14) + 'px';
      tooltip.style.top  = (my - 10) + 'px';
      tooltip.classList.remove('hidden');
    } else {
      tooltip.classList.add('hidden');
    }
  });
  canvas.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));

  // Resize
  new ResizeObserver(() => {
    canvas.width  = wrap.clientWidth  || 800;
    canvas.height = wrap.clientHeight || 500;
    draw();
  }).observe(wrap);
}

/* ── Script loader ───────────────────────────────────────────────────────── */
function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const s   = document.createElement('script');
    s.src     = src;
    s.onload  = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}
