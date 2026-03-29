// --- Internal Config & State ---
// Default labels match Legacy ISM band (433 MHz → ~5.8 GHz), synced from backend on connect
let freqLabels = ['433M', '868M', '915M', '1.2G', '2.4G', '3.5G', '4.9G', '5.2G', '5.5G', '5.8G'];
let currentViewMode = 'spectrum'; // 'spectrum' | 'waterfall' | 'grid'
let waterfallHistory = [];
const HISTORY_LEN = 80;
let freqChart = null;
let lossChart = null;
let lastChartUpdate = 0;

// ─── Channel Grid State ────────────────────────────────────────────────────
const NUM_CHANNELS   = 10;
const GRID_COLS      = 5;   // 5 columns × 2 rows
/** @type {HTMLCanvasElement[]} */
let chCanvases       = [];  // 10 mini canvas elements
/** @type {CanvasRenderingContext2D[]} */
let chCtxs           = [];  // 10 2D contexts
/** @type {number[][]} */
let chHistory        = [];  // per-channel rolling power history (for waterfall bar)
const CH_HISTORY_LEN = 30;  // frames to keep per mini-graph
let lastActiveChIdx  = -1;  // last highlighted channel index
let lastGridUpdate   = 0;   // throttle grid highlight updates
// ──────────────────────────────────────────────────────────────────────────

// --- DOM Elements ---
const latVal = document.getElementById('latency-val');
const chVal = document.getElementById('channel-val');
const rVal = document.getElementById('reward-val'); 
const hoVal = document.getElementById('handover-val');
const pwrVal = document.getElementById('power-val');
const stepVal = document.getElementById('steps-val');
const rLog = document.getElementById('reasoning-log');
const confSlider = document.getElementById('conf-slider');
const pwrSlider = document.getElementById('pwr-slider');
const confDisp = document.getElementById('conf-val-display');
const pwrDisp = document.getElementById('pwr-val-display');
const predictDisp = document.getElementById('predict-val');
const predictBar = document.getElementById('predict-bar');
const priorityDisp = document.getElementById('priority-val');
const ecoVal = document.getElementById('eco-val');
const ecoHopVal = document.getElementById('eco-hop-val');
const ecoBar = document.getElementById('eco-bar');
const plsVal = document.getElementById('pls-val');

// New DOM elements
const predConfVal = document.getElementById('pred-conf-val');
const trendIcon = document.getElementById('trend-icon');
const tl100 = document.getElementById('tl-100');
const tl200 = document.getElementById('tl-200');
const tl500 = document.getElementById('tl-500');

// AI Config Sliders
const aiThreshSlider = document.getElementById('ai-thresh-slider');
const aiThreshDisp = document.getElementById('ai-thresh-display');
const histWSlider = document.getElementById('hist-w-slider');
const histWDisp = document.getElementById('hist-w-display');
const uncertWSlider = document.getElementById('uncert-w-slider');
const uncertWDisp = document.getElementById('uncert-w-display');

// Chart instances
let accChart = null;
let predictionHistory = [];
let actualHistory = [];
const CHART_MAX_POINTS = 50;

const mimoCanvas = document.getElementById('mimo-canvas');
const radarCanvas = document.getElementById('radar-canvas');
const radarCtx = radarCanvas?.getContext('2d');
const mimoCtx = mimoCanvas?.getContext('2d');
let targetCache = new Map(); // Persistent target memory {id: {dist, angle, lastSeen}}
let scanAngle = 0; // Current sweep angle
const classNameVal = document.getElementById('class-name-val');
const confVal = document.getElementById('conf-val');
const viewToggleBtn = document.getElementById('view-toggle-btn');
const viewModeIcon = document.getElementById('view-mode-icon');
const canvas = document.getElementById('spectrum-canvas');
const ctx = canvas?.getContext('2d');
const alertCollision = document.getElementById('alert-collision');
const alertHandover = document.getElementById('alert-handover');

// --- Initialization ---
let learningSteps = parseInt(stepVal?.textContent) || 0;

function setupLayerTabs() {
    const tabs = document.querySelectorAll('.layer-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const layer = tab.getAttribute('data-layer');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'switch_layer', layer: layer }));
                
                // Optimistic UI update
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
            }
        });
    });
}

function setupTabs() {
    // Layer tabs are already in HTML - skip DOM rebuild to preserve layer-tab handlers
    const container = document.querySelector('.frequency-tabs');
    if (!container) return;
    // Don't clear - layer tabs are already rendered from index.html
    // setupLayerTabs() handles their click events separately
}

// ─── Frequency Channel Cards: individual divs + canvases ──────────────────────
/** @type {HTMLCanvasElement[]} */
let fchCanvases = [];
/** @type {CanvasRenderingContext2D[]} */
let fchCtxs     = [];
/** @type {number[][]} */
let fchQHistory  = Array.from({length: 10}, () => []);
const FCH_HIST   = 28;   // frames of Q-history per mini-graph

function initFreqMatrix() {   // kept same name so existing call in load event works
    const grid = document.getElementById('freq-ch-grid');
    if (!grid) return;

    grid.innerHTML = '';
    fchCanvases = [];
    fchCtxs     = [];
    fchQHistory = Array.from({length: 10}, () => []);

    for (let i = 0; i < 10; i++) {
        const card = document.createElement('div');
        card.className = 'fch-card';
        card.id = `fch-card-${i}`;

        const strip = document.createElement('div');
        strip.className = 'fch-busy-strip';

        const label = document.createElement('div');
        label.className = 'fch-label';
        label.id = `fch-label-${i}`;
        label.textContent = `CH${i}`;

        const cv = document.createElement('canvas');
        cv.className = 'fch-canvas';
        cv.id = `fch-cv-${i}`;
        cv.width  = 80;
        cv.height = 56;

        const qval = document.createElement('div');
        qval.className = 'fch-qval';
        qval.id = `fch-qv-${i}`;
        qval.textContent = 'Q:---';

        card.appendChild(strip);
        card.appendChild(label);
        card.appendChild(cv);
        card.appendChild(qval);
        grid.appendChild(card);

        // Click → manual channel override
        const idx = i;
        card.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'set_channel', idx }));
            }
        });

        fchCanvases.push(cv);
        fchCtxs.push(cv.getContext('2d'));
    }
}

/**
 * Draws a single channel card: Q-value history line + current bar.
 * @param {number}   i        - channel index 0-9
 * @param {number[]} qHist    - rolling Q-value history
 * @param {number}   qNow     - current Q-value (raw, can be negative)
 * @param {boolean}  isActive
 * @param {boolean}  isBusy
 */
function drawFchCard(i, qHist, qNow, isActive, isBusy) {
    const cv   = fchCanvases[i];
    const ctx2 = fchCtxs[i];
    if (!cv || !ctx2) return;

    const w = cv.width;
    const h = cv.height;
    ctx2.clearRect(0, 0, w, h);

    // Background
    ctx2.fillStyle = isActive ? 'rgba(34,211,238,0.05)' : 'rgba(0,0,0,0.3)';
    ctx2.fillRect(0, 0, w, h);

    if (qHist.length < 2) return;

    // Normalise Q-values: map range [min, max] → [0, 1]
    const allVals  = [...qHist, qNow];
    const qMin     = Math.min(...allVals);
    const qMax     = Math.max(...allVals);
    const qRange   = (qMax - qMin) || 1;
    const norm     = v => Math.max(0, Math.min(1, (v - qMin) / qRange));

    const accent = isActive ? '#22d3ee' : isBusy ? '#ef4444' : '#38bdf8';
    const alpha  = isActive ? 0.22 : 0.10;

    // Subtle grid line
    ctx2.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx2.lineWidth = 0.5;
    ctx2.beginPath(); ctx2.moveTo(0, h/2); ctx2.lineTo(w, h/2); ctx2.stroke();

    // Area fill under Q-history line
    const n     = qHist.length;
    const stepX = (w - 2) / Math.max(n - 1, 1);
    const grad  = ctx2.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, accent.replace('#22d3ee', `rgba(34,211,238,${alpha})`)
                               .replace('#ef4444', `rgba(239,68,68,${alpha})`)
                               .replace('#38bdf8', `rgba(56,189,248,${alpha})`));
    grad.addColorStop(1, 'rgba(0,0,0,0)');

    ctx2.beginPath();
    ctx2.moveTo(1, h);
    qHist.forEach((v, idx2) => ctx2.lineTo(1 + idx2 * stepX, h - norm(v) * (h - 4) - 2));
    ctx2.lineTo(1 + (n-1) * stepX, h);
    ctx2.closePath();
    ctx2.fillStyle = grad;
    ctx2.fill();

    // Q-history stroke line
    ctx2.beginPath();
    qHist.forEach((v, idx2) => {
        const x = 1 + idx2 * stepX;
        const y = h - norm(v) * (h - 4) - 2;
        idx2 === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
    });
    ctx2.strokeStyle = accent;
    ctx2.lineWidth   = isActive ? 1.5 : 1;
    if (isActive) { ctx2.shadowBlur = 5; ctx2.shadowColor = '#22d3ee'; }
    ctx2.stroke();
    ctx2.shadowBlur = 0;

    // Right-edge current-value bar  
    const barH  = norm(qNow) * (h - 4);
    ctx2.fillStyle = accent;
    ctx2.globalAlpha = isActive ? 0.85 : 0.45;
    ctx2.fillRect(w - 5, h - barH - 2, 4, barH);
    ctx2.globalAlpha = 1;
}

/**
 * Master update: called from ws.onmessage with fresh telemetry.
 * @param {number[]} qValues    - 10+ Q-values from D3QN agent
 * @param {number}   activeIdx  - currently tuned channel index
 * @param {boolean}  isBusy     - is active channel occupied?
 */
function updateFreqChannels(qValues, activeIdx, isBusy) {
    const grid = document.getElementById('freq-ch-grid');
    if (!grid || !qValues || qValues.length === 0) return;

    grid.classList.toggle('has-active', activeIdx >= 0);

    for (let i = 0; i < 10; i++) {
        const q = qValues[i] ?? 0;

        // Update rolling history
        fchQHistory[i].push(q);
        if (fchQHistory[i].length > FCH_HIST) fchQHistory[i].shift();

        const isActive = (i === activeIdx);
        const chBusy   = isActive && isBusy;

        // Draw canvas
        drawFchCard(i, fchQHistory[i], q, isActive, chBusy);

        // Update Q-value text
        const qv = document.getElementById(`fch-qv-${i}`);
        if (qv) qv.textContent = `Q:${q.toFixed(2)}`;

        // Update label from freqLabels (may have changed on layer switch)
        const lbl = document.getElementById(`fch-label-${i}`);
        if (lbl) lbl.textContent = `CH${i}: ${freqLabels[i] || '---'}`;

        // Toggle CSS classes
        const card = document.getElementById(`fch-card-${i}`);
        if (card) {
            card.classList.toggle('fch-active', isActive);
            card.classList.toggle('fch-busy',   chBusy);
        }
    }

    // Update active label in panel header
    const lbl = document.getElementById('fch-active-label');
    if (lbl) {
        lbl.textContent = activeIdx >= 0
            ? `ACTIVE ► CH${activeIdx}: ${freqLabels[activeIdx] || '?'}`
            : '';
    }
}

function initLossChart() {
    const lossCtx = document.getElementById('loss-canvas')?.getContext('2d');
    if(lossCtx) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 20}, (_, i) => i + 1),
                datasets: [{
                    label: 'Agent Reward',
                    data: new Array(20).fill(0),
                    borderColor: '#ec4899',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointBackgroundColor: '#ec4899',
                    tension: 0.3,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { 
                        display: true, 
                        position: 'right',
                        min: -100, 
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: 'rgba(255,255,255,0.2)', font: { size: 8 } }
                    }
                },
                plugins: { legend: { display: false } },
                layout: { padding: { left: 5, right: 30, top: 15, bottom: 5 } }
            }
        });
        logToTerminal("AI Engine Graph Initialized.", "success");
    } else {
        logToTerminal("CRITICAL: AI Graph Canvas NOT found!", "warn");
    }
}

function updateTabUI(visualIdx) {
    // Only update non-layer-tab buttons (avoid conflicting with layer active state)
    const tabs = document.querySelectorAll('.frequency-tabs .tab:not(.layer-tab)');
    tabs.forEach((t, i) => {
        if(i === visualIdx) t.classList.add('active');
        else t.classList.remove('active');
    });
}

function resizeCanvas() {
    if(canvas && canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
    }
}
window.addEventListener('load', () => {
    resizeCanvas();
    setupTabs();
    initFreqMatrix();
    initLossChart();
    initMIMO();
    initRadar();
    initFrequencyGrid();   // ← Channel grid
    setupLayerTabs();
});

// ═══════════════════════════════════════════════════════════════════════════
// FREQUENCY CHANNEL GRID
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build 10 mini-canvas cards inside #freq-grid-container.
 * Called once on page load. Cards persist; only content is redrawn.
 */
function initFrequencyGrid() {
    const container = document.getElementById('freq-grid-container');
    if (!container) return;

    // Clear any previous build (e.g. from resize)
    container.innerHTML = '';
    chCanvases = [];
    chCtxs     = [];
    chHistory  = Array.from({ length: NUM_CHANNELS }, () => []);

    for (let i = 0; i < NUM_CHANNELS; i++) {
        // Card wrapper
        const card = document.createElement('div');
        card.className = 'frequency-grid-item';
        card.id = `ch-card-${i}`;

        // Busy indicator strip
        const strip = document.createElement('div');
        strip.className = 'ch-busy-strip';

        // Label  e.g. "CH0: 433M"
        const label = document.createElement('div');
        label.className = 'ch-label';
        label.id = `ch-label-${i}`;
        label.textContent = `CH${i}: ${freqLabels[i] || '---'}`;

        // Mini canvas
        const cv = document.createElement('canvas');
        cv.className = 'ch-canvas';
        cv.id = `ch-canvas-${i}`;
        // Pixel dimensions — set after append so clientWidth works
        cv.width  = 100;
        cv.height = 52;

        // Power badge
        const badge = document.createElement('div');
        badge.className = 'ch-power-badge';
        badge.id = `ch-badge-${i}`;
        badge.textContent = '---';

        card.appendChild(strip);
        card.appendChild(label);
        card.appendChild(cv);
        card.appendChild(badge);
        container.appendChild(card);

        // Wire click → manual channel override
        const chIdx = i;
        card.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ cmd: 'set_channel', idx: chIdx }));
            }
        });

        chCanvases.push(cv);
        const cx = cv.getContext('2d');
        chCtxs.push(cx);
    }
}

/**
 * Refresh labels when freqLabels array changes (layer switch).
 */
function syncGridLabels() {
    for (let i = 0; i < NUM_CHANNELS; i++) {
        const el = document.getElementById(`ch-label-${i}`);
        if (el) el.textContent = `CH${i}: ${freqLabels[i] || '---'}`;
    }
}

/**
 * Draws mini power-bar waveform for one channel.
 * @param {number}   chIdx   - 0..9
 * @param {number[]} history - recent power values (dB)
 * @param {boolean}  isActive
 * @param {boolean}  isBusy
 */
function drawChannelMiniGraph(chIdx, history, isActive, isBusy) {
    const cv  = chCanvases[chIdx];
    const ctx2 = chCtxs[chIdx];
    if (!cv || !ctx2) return;

    const w = cv.width;
    const h = cv.height;
    ctx2.clearRect(0, 0, w, h);

    if (history.length === 0) return;

    // Background fill
    ctx2.fillStyle = isActive
        ? 'rgba(34, 211, 238, 0.04)'
        : 'rgba(0, 0, 0, 0.25)';
    ctx2.fillRect(0, 0, w, h);

    const accentColor = isActive ? '#22d3ee' : isBusy ? '#ef4444' : '#38bdf8';
    const fillAlpha   = isActive ? 0.25 : 0.10;

    // Normalize: map [-100, 20] dB range → 0..1
    const normalize = v => Math.max(0, Math.min(1, (v + 100) / 120));

    const n   = history.length;
    const stepX = w / Math.max(n - 1, 1);

    // Area fill
    const grad = ctx2.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, accentColor.replace(')', `, ${fillAlpha})`).replace('rgb', 'rgba').replace('#22d3ee', `rgba(34,211,238,${fillAlpha})`).replace('#ef4444', `rgba(239,68,68,${fillAlpha})`).replace('#38bdf8', `rgba(56,189,248,${fillAlpha})`) );
    grad.addColorStop(1, 'rgba(0,0,0,0)');

    ctx2.beginPath();
    ctx2.moveTo(0, h);
    history.forEach((val, i) => {
        const x = i * stepX;
        const y = h - normalize(val) * h;
        ctx2.lineTo(x, y);
    });
    ctx2.lineTo((n - 1) * stepX, h);
    ctx2.closePath();
    ctx2.fillStyle = grad;
    ctx2.fill();

    // Stroke line
    ctx2.beginPath();
    history.forEach((val, i) => {
        const x = i * stepX;
        const y = h - normalize(val) * h;
        if (i === 0) ctx2.moveTo(x, y);
        else ctx2.lineTo(x, y);
    });
    ctx2.strokeStyle = accentColor;
    ctx2.lineWidth = isActive ? 1.5 : 1;
    if (isActive) {
        ctx2.shadowBlur    = 6;
        ctx2.shadowColor   = '#22d3ee';
    }
    ctx2.stroke();
    ctx2.shadowBlur = 0;

    // Peak tick mark
    const peak = Math.max(...history);
    const peakY = h - normalize(peak) * h;
    ctx2.fillStyle = accentColor;
    ctx2.fillRect(w - 3, peakY - 1, 3, 2);
}

/**
 * Main grid render call — invoked from ws.onmessage.
 * @param {number[]} fullSlice   - waterfall_slice power array (1024 bins)
 * @param {number}   activeIdx   - current channel index from backend
 * @param {boolean}  isBusy      - is active channel busy?
 */
function drawFrequencyGrid(fullSlice, activeIdx, isBusy) {
    if (!fullSlice || fullSlice.length === 0) return;

    const sliceLen  = fullSlice.length;
    const binsPerCh = Math.floor(sliceLen / NUM_CHANNELS);

    for (let i = 0; i < NUM_CHANNELS; i++) {
        // Compute average power in this channel's bin slice
        const start = i * binsPerCh;
        const end   = start + binsPerCh;
        const binSlice = fullSlice.slice(start, end);
        const avgPwr   = binSlice.reduce((a, b) => a + b, 0) / binSlice.length;

        // Push to history
        chHistory[i].push(avgPwr);
        if (chHistory[i].length > CH_HISTORY_LEN) chHistory[i].shift();

        const isActive = (i === activeIdx);
        const chBusy   = isActive && isBusy;

        drawChannelMiniGraph(i, chHistory[i], isActive, chBusy);

        // Sync badge power value
        const badge = document.getElementById(`ch-badge-${i}`);
        if (badge) badge.textContent = `${avgPwr.toFixed(0)}dB`;
    }

    // Highlighting — throttle to ~10Hz
    const now = Date.now();
    if (now - lastGridUpdate > 100 || activeIdx !== lastActiveChIdx) {
        updateChannelHighlighting(activeIdx, isBusy);
        lastActiveChIdx = activeIdx;
        lastGridUpdate  = now;
    }

    // Keep labels in sync with current layer
    syncGridLabels();
}

/**
 * Applies/removes CSS classes for active glow and dim effect.
 */
function updateChannelHighlighting(activeIdx, isBusy) {
    const container = document.getElementById('freq-grid-container');
    if (!container) return;

    // Add has-active on container to trigger dim-of-non-active via CSS
    container.classList.toggle('has-active', activeIdx >= 0);

    for (let i = 0; i < NUM_CHANNELS; i++) {
        const card = document.getElementById(`ch-card-${i}`);
        if (!card) continue;
        card.classList.toggle('channel-active', i === activeIdx);
        card.classList.toggle('ch-is-busy', i === activeIdx && isBusy);
    }
}

/**
 * Switch between 'spectrum', 'waterfall', and 'grid' view modes.
 * Handles panel class toggling and icon updates.
 */
function setViewMode(mode) {
    currentViewMode = mode;
    const psdPanel   = document.querySelector('.psd-panel');
    const titleEl    = document.getElementById('psd-panel-title');
    const iconEl     = document.getElementById('view-mode-icon');
    const gridBtn    = document.getElementById('grid-toggle-btn');

    if (mode === 'grid') {
        psdPanel?.classList.add('grid-mode');
        if (titleEl) titleEl.textContent = 'FREQUENCY CHANNEL GRID';
        if (gridBtn) gridBtn.style.color = '#22d3ee';
        if (iconEl)  iconEl.textContent  = '📊'; // spectrum icon (clicking goes back)
        waterfallHistory = [];
    } else {
        psdPanel?.classList.remove('grid-mode');
        if (titleEl) titleEl.textContent = 'SPECTRUM INTENSITY (PSD)';
        if (gridBtn) gridBtn.style.color = '';
        if (iconEl)  iconEl.textContent  = mode === 'waterfall' ? '🌊' : '📊';
    }
}

function initRadar() {
    if(!radarCanvas) return;
    const resize = () => {
        radarCanvas.width = radarCanvas.clientWidth;
        radarCanvas.height = radarCanvas.clientHeight;
    };
    window.addEventListener('resize', resize);
    resize();
    setupLayerTabs();
}

function drawRadar(objects) {
    if(!radarCtx) return;
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    const cx = w/2;
    const cy = h/2;
    const radius = Math.min(cx, cy) - 10;
    const maxDist = 100;
    const now = Date.now();

    // Echo Persistence Effect: Fade instead of clear
    radarCtx.fillStyle = 'rgba(11, 17, 32, 0.15)'; 
    radarCtx.fillRect(0,0,w,h);
    
    // Increment scan angle (Internal logic)
    scanAngle = (scanAngle + 3) % 360; // Slightly faster sweep for better UX
    
    // Sync the visual (neon) scan line element if it exists
    const scanLineEl = document.querySelector('.radar-scan-line');
    if (scanLineEl) {
        scanLineEl.style.transform = `rotate(${scanAngle}deg)`;
    }
    
    // Draw grid
    radarCtx.strokeStyle = 'rgba(34, 211, 238, 0.1)';
    radarCtx.lineWidth = 1;
    radarCtx.beginPath();
    [0.3, 0.6, 1.0].forEach(r => {
        radarCtx.moveTo(cx + radius*r, cy);
        radarCtx.arc(cx, cy, radius*r, 0, Math.PI*2);
    });
    radarCtx.stroke();
    
    // Draw canvas scan line for correlation accuracy
    const angleRadLine = (scanAngle - 90) * Math.PI/180;
    const scanLineX = cx + radius * Math.cos(angleRadLine);
    const scanLineY = cy + radius * Math.sin(angleRadLine);
    
    radarCtx.strokeStyle = 'rgba(34, 211, 238, 0.3)';
    radarCtx.beginPath();
    radarCtx.moveTo(cx, cy);
    radarCtx.lineTo(scanLineX, scanLineY);
    radarCtx.stroke();

    // Update Target Cache with new telemetry data
    if(objects && Array.isArray(objects)) {
        objects.forEach(obj => {
            const diff = Math.abs(scanAngle - obj.angle);
            const isHit = diff < 15 || diff > 345; // 30-degree hit window
            
            let entry = targetCache.get(obj.id) || { dist: obj.dist, angle: obj.angle, lastSeen: 0 };
            entry.dist = obj.dist;
            entry.angle = obj.angle;
            
            if (isHit) {
                entry.lastSeen = now;
            }
            targetCache.set(obj.id, entry);
        });
    }

    // Render Cached Targets
    targetCache.forEach((entry, id) => {
        const timeSinceSeen = now - entry.lastSeen;
        if (timeSinceSeen > 4000) { // Timeout after 1 full sweep
            targetCache.delete(id);
            return;
        }

        // Target opacity based on recency
        const opacity = Math.max(0, 1 - (timeSinceSeen / 4000));
        const angleRad = (entry.angle - 90) * (Math.PI/180);
        const r = (entry.dist / maxDist) * radius;
        const x = cx + r * Math.cos(angleRad);
        const y = cy + r * Math.sin(angleRad);
        
        // Draw target blip
        radarCtx.fillStyle = `rgba(239, 68, 68, ${opacity})`;
        radarCtx.shadowBlur = opacity * 15;
        radarCtx.shadowColor = '#ef4444';
        radarCtx.beginPath();
        radarCtx.arc(x, y, 4, 0, Math.PI*2);
        radarCtx.fill();
        radarCtx.shadowBlur = 0;
        
        // Target Box & Labels (Only if recently hit)
        if (timeSinceSeen < 500) {
            radarCtx.strokeStyle = `rgba(239, 68, 68, ${opacity * 0.5})`;
            radarCtx.strokeRect(x - 8, y - 8, 16, 16);
            radarCtx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.8})`;
            radarCtx.font = '9px Orbitron';
            radarCtx.fillText(`OBJ-${id} (${entry.dist.toFixed(0)}m)`, x + 10, y + 3);
        }
    });
}

function initMIMO() {
    if(!mimoCanvas) return;
    mimoCanvas.width = 100;
    mimoCanvas.height = 100;
    drawMIMO(0);
}

function drawMIMO(activeBeam) {
    if(!mimoCtx) return;
    const w = mimoCanvas.width;
    const h = mimoCanvas.height;
    const cx = w/2;
    const cy = h/2;
    
    mimoCtx.clearRect(0, 0, w, h);
    
    // Draw base circles
    mimoCtx.strokeStyle = 'rgba(34, 211, 238, 0.2)';
    mimoCtx.beginPath();
    mimoCtx.arc(cx, cy, 40, 0, Math.PI*2);
    mimoCtx.stroke();
    
    // Draw 4 beam sectors
    const angles = [0, Math.PI/2, Math.PI, 3*Math.PI/2];
    angles.forEach((angle, i) => {
        const isSelected = i === activeBeam;
        mimoCtx.fillStyle = isSelected ? '#22d3ee' : 'rgba(34, 211, 238, 0.1)';
        mimoCtx.beginPath();
        mimoCtx.moveTo(cx, cy);
        mimoCtx.arc(cx, cy, isSelected ? 45 : 35, angle - 0.4, angle + 0.4);
        mimoCtx.closePath();
        mimoCtx.fill();
        
        if(isSelected) {
            mimoCtx.shadowBlur = 10;
            mimoCtx.shadowColor = '#22d3ee';
            mimoCtx.stroke();
            mimoCtx.shadowBlur = 0;
        }
    });
}
window.addEventListener('resize', resizeCanvas);

// --- Drawing Functions ---
function drawGrid(w, h) {
    if(!ctx) return;
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.05)';
    ctx.lineWidth = 0.5;
    
    // Vertical Grid
    for(let x=0; x<=w; x += w/10) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    // Horizontal Grid
    for(let y=0; y<=h; y += h/6) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
}

function drawSpectrum(slice) {
    if(!ctx || !slice || slice.length === 0) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    
    drawGrid(w, h);
    
    const numBins = slice.length;
    const stepX = w / (numBins - 1);
    
    // Area Gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, 'rgba(34, 211, 238, 0.15)');
    gradient.addColorStop(1, 'rgba(34, 211, 238, 0.0)');
    
    ctx.beginPath();
    ctx.moveTo(0, h);
    for(let i=0; i<numBins; i++) {
        const normalized = Math.max(0, Math.min(1, (slice[i] + 100) / 120));
        ctx.lineTo(i * stepX, h - (normalized * h));
    }
    ctx.lineTo(w, h);
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Glowing Line
    ctx.save();
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#22d3ee';
    ctx.beginPath();
    for(let i=0; i<numBins; i++) {
        const normalized = Math.max(0, Math.min(1, (slice[i] + 100) / 120));
        const x = i * stepX;
        const y = h - (normalized * h);
        if(i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#22d3ee';
    ctx.stroke();
    ctx.restore();
}

function drawWaterfall(slice) {
    if(!ctx || !slice || slice.length === 0) return;
    waterfallHistory.unshift(slice);
    if(waterfallHistory.length > HISTORY_LEN) waterfallHistory.pop();
    
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    
    const numBins = slice.length;
    const cellW = w / numBins;
    const cellH = h / HISTORY_LEN;
    
    for(let i=0; i<waterfallHistory.length; i++) {
        const rowData = waterfallHistory[i];
        const y = i * cellH;
        for(let j=0; j<numBins; j++) {
            const val = rowData[j];
            const normalized = Math.max(0, Math.min(1, (val + 90) / 100)); 
            if(normalized < 0.1) continue;
            
            // High-contrast Plasma/Inferno palette
            const hue = 280 - (normalized * 280); // Purple -> Blue -> Cyan -> Green -> Yellow -> Red
            ctx.fillStyle = `hsla(${hue}, 80%, ${30 + normalized * 40}%, ${normalized * 0.8})`;
            ctx.fillRect(j * cellW, y, Math.ceil(cellW)+1, Math.ceil(cellH)+1);
        }
    }
}

function logToTerminal(msg, type='normal') {
    if(!rLog) return;
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    const timestamp = new Date().toLocaleTimeString('en-GB', { hour12: false });
    line.innerHTML = `<span style="color:var(--text-muted)">[${timestamp}]</span> ${msg}`;
    rLog.appendChild(line);
    rLog.scrollTop = rLog.scrollHeight;
    if(rLog.childNodes.length > 500) rLog.removeChild(rLog.firstChild);
}

// --- WebSocket Connection ---
const ws = new WebSocket(`ws://${window.location.host}/ws`);
window.ws = ws;

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.sdr_linked === false) {
        document.getElementById('freeze-overlay').style.display = 'flex';
        return;
    } else {
        document.getElementById('freeze-overlay').style.display = 'none';
    }
    
    if(latVal) latVal.textContent = `${data.latency_ms} ms`;
    if(chVal) chVal.textContent = freqLabels[data.channel_idx] || "N/A";
    
    // Sync Frequency Labels from Backend
    if (data.layer_labels && data.layer_labels.length > 0) {
        freqLabels = data.layer_labels;
    }

    // Sync Active Layer Tab
    document.querySelectorAll('.layer-tab').forEach(tab => {
        if (tab.getAttribute('data-layer') === data.current_layer) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    if(rVal) {
        const lossVal = data.d3qn_loss !== undefined ? data.d3qn_loss : 0.0;
        rVal.textContent = lossVal.toFixed(3);
    }
    if(pwrVal && data.waterfall_slice.length > 0) {
        const peak = Math.max(...data.waterfall_slice);
        pwrVal.textContent = `${peak.toFixed(1)} dBm`;
    }
    
    // Update AI Engine Graph (Loss/Reward Chart)
    if(lossChart) {
        const val = (data.d3qn_loss !== undefined && !isNaN(data.d3qn_loss)) ? data.d3qn_loss : 0;
        const chartData = lossChart.data.datasets[0].data;
        chartData.push(val);
        if(chartData.length > 20) chartData.shift();
        
        const now = Date.now();
        if(now - lastChartUpdate > 100) { // Throttle to 10Hz
            lossChart.update('none');
            lastChartUpdate = now;
        }
    } else if (data.reward !== undefined) {
        console.warn("Telemetry arriving but lossChart not ready.");
    }
    
    updateTabUI(Math.floor(data.channel_idx / 2));

    if (data.reasoning_msg) {
        const type = data.is_busy ? 'warn' : 'success';
        logToTerminal(data.reasoning_msg, type);
    }

    // Sync Mode Tag
    const modeToggle = document.getElementById('mode-toggle');
    if (modeToggle) {
        if (data.manual_mode) {
            modeToggle.textContent = 'MANUAL';
            modeToggle.className = 'mode-tag manual';
        } else {
            modeToggle.textContent = 'AUTO';
            modeToggle.className = 'mode-tag auto';
        }
    }

    // Sync Pause/Resume State
    const pauseBtn = document.getElementById('pause-btn');
    const haltOverlay = document.getElementById('halted-overlay');
    if (pauseBtn) {
        pauseBtn.querySelector('span').textContent = data.is_paused ? 'RESUME SCAN' : 'STOP SCAN';
        if (data.is_paused) pauseBtn.classList.add('paused');
        else pauseBtn.classList.remove('paused');
    }
    if (haltOverlay) {
        haltOverlay.style.display = data.is_paused ? 'flex' : 'none';
    }

    if (data.event_trigger === 'collision') {
        alertCollision?.play().catch(()=>{});
        learningSteps++;
        if(stepVal) stepVal.textContent = learningSteps;
        if(hoVal) hoVal.textContent = `${data.latency_ms} ms`;
    } else {
        if(hoVal) hoVal.textContent = `0.00ms`;
    }
    
    const classColors = {
        'Idle': '#10b981',
        '6G-URLLC': '#ef4444',
        '6G-eMBB': '#22d3ee',
        'Legacy IoT': '#fbbf24'
    };
    const activeColor = classColors[data.class_name] || '#6366f1';
    
    if(classNameVal) {
        classNameVal.textContent = data.class_name;
        classNameVal.style.color = activeColor;
    }
    
    if(priorityDisp) {
        priorityDisp.textContent = data.priority > 0.8 ? 'CRITICAL (URLLC)' : 
                                   data.priority > 0.4 ? 'HIGH (eMBB)' : 
                                   data.priority > 0 ? 'STANDARD (IoT)' : 'NONE';
        priorityDisp.style.color = activeColor;
    }
    
    if(predictDisp && predictBar) {
        const pStr = (data.prediction_horizon * 100).toFixed(0) + '%';
        predictDisp.textContent = pStr;
        // Dynamic Gradient for predictBar (Green to Red based on prediction 0->1)
        const green = Math.max(0, 255 * (1 - data.prediction_horizon));
        const red = Math.max(0, 255 * data.prediction_horizon);
        predictBar.style.width = pStr;
        predictBar.style.background = `linear-gradient(90deg, rgba(16,185,129,0.8) 0%, rgba(${red},${green},50,0.8) 100%)`;
        
        // Multi-Horizon Timelines
        if (data.forecast_array) {
            if(tl100) tl100.style.width = (data.forecast_array[0] * 100).toFixed(0) + '%';
            if(tl200) tl200.style.width = (data.forecast_array[1] * 100).toFixed(0) + '%';
            if(tl500) tl500.style.width = (data.forecast_array[2] * 100).toFixed(0) + '%';
            
            // Timeline colors
            [tl100, tl200, tl500].forEach((el, i) => {
                if(el) el.style.backgroundColor = data.forecast_array[i] > 0.8 ? '#ef4444' : (data.forecast_array[i] > 0.4 ? '#fbbf24' : '#10b981');
            });
        }
        
        if (predConfVal && data.prediction_confidence !== undefined) {
            predConfVal.textContent = (data.prediction_confidence * 100).toFixed(1) + '%';
        }
        if (trendIcon && data.trend) {
            trendIcon.textContent = data.trend === 'rising' ? '⬆️' : data.trend === 'falling' ? '⬇️' : '➡️';
        }
    }
    
    // Update Accuracy Chart
    if (accChart) {
        predictionHistory.push(data.prediction_horizon);
        actualHistory.push(data.is_busy ? 1.0 : 0.0);
        if (predictionHistory.length > CHART_MAX_POINTS) {
            predictionHistory.shift();
            actualHistory.shift();
        }
        accChart.data.datasets[0].data = predictionHistory;
        accChart.data.datasets[1].data = actualHistory;
        accChart.update('none'); // Update without animation for performance
    }

    drawMIMO(data.active_beam);
    drawRadar(data.radar_map);

    if(ecoVal && ecoBar) {
        const ecoStr = data.eco_saving + '%';
        ecoVal.textContent = ecoStr;
        ecoBar.style.width = ecoStr;
        
        if (ecoHopVal && data.last_hop_db) {
            const dbVal = parseFloat(data.last_hop_db);
            ecoHopVal.textContent = data.last_hop_db + ' dB';
            // Color code: green for positive (saving), red for negative (cost)
            ecoHopVal.style.color = dbVal > 0 ? '#10b981' : dbVal < 0 ? '#ef4444' : 'var(--accent-cyan)';
        }
    }
    
    if(plsVal) {
        plsVal.textContent = data.pls_score + '%';
        plsVal.style.color = data.pls_score > 80 ? '#10b981' : data.pls_score > 40 ? '#fbbf24' : '#ef4444';
        const plsBar = document.getElementById('pls-bar');
        if (plsBar) {
            plsBar.style.width = data.pls_score + '%';
            plsBar.style.background = data.pls_score > 80 ? '#10b981' : data.pls_score > 40 ? '#fbbf24' : '#ef4444';
        }
    }

    if(confVal) confVal.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    
    if (currentViewMode === 'spectrum') drawSpectrum(data.waterfall_slice);
    else if (currentViewMode === 'waterfall') drawWaterfall(data.waterfall_slice);
    else if (currentViewMode === 'grid')     drawFrequencyGrid(data.waterfall_slice, data.channel_idx, data.is_busy);
    
    if (data.q_values && data.q_values.length > 0) {
        updateFreqChannels(data.q_values, data.channel_idx, data.is_busy);
    }
};

// --- Event Listeners ---
confSlider?.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if(confDisp) confDisp.textContent = val.toFixed(2);
    ws.send(JSON.stringify({ cmd: 'update_params', conf: val }));
});
pwrSlider?.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    if(pwrDisp) pwrDisp.textContent = `${val} dB`;
    ws.send(JSON.stringify({ cmd: 'update_params', pwr: val }));
});

viewToggleBtn?.addEventListener('click', () => {
    // Spectrum ↔ Waterfall (only when not in grid mode)
    if (currentViewMode === 'grid') return; // grid-toggle-btn handles exit
    const next = currentViewMode === 'spectrum' ? 'waterfall' : 'spectrum';
    setViewMode(next);
    waterfallHistory = [];
});

document.getElementById('grid-toggle-btn')?.addEventListener('click', () => {
    const next = currentViewMode === 'grid' ? 'spectrum' : 'grid';
    setViewMode(next);
});

document.getElementById('view-more-btn')?.addEventListener('click', function() {
    this.classList.toggle('open');
    document.getElementById('expanded-freq-panel')?.classList.toggle('open');
});

// Mode Toggle Listener
document.getElementById('mode-toggle')?.addEventListener('click', function() {
    const isManual = this.classList.contains('manual');
    ws.send(JSON.stringify({ cmd: 'toggle_manual_mode', val: !isManual }));
});

// Pause/Resume Listener
document.getElementById('pause-btn')?.addEventListener('click', function() {
    const isPaused = this.classList.contains('paused');
    ws.send(JSON.stringify({ cmd: 'toggle_pause', val: !isPaused }));
});

document.getElementById('exit-btn')?.addEventListener('click', () => {
    if(confirm("Terminate AEGIS hardware loop?")) {
        ws.send(JSON.stringify({ cmd: 'exit' }));
        document.body.innerHTML = '<div style="background:#0b1120;color:#ef4444;height:100vh;display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:32px;">SYSTEM TERMINATED</div>';
    }
});

// Terminal Minimize Toggle
document.getElementById('log-minimize-btn')?.addEventListener('click', () => {
    document.getElementById('terminal-drawer')?.classList.toggle('collapsed');
});

// Reset HUD Button
document.getElementById('reset-btn')?.addEventListener('click', () => {
    // Reset all stats displays to defaults
    if (latVal)       latVal.textContent  = '0 ms';
    if (hoVal)        hoVal.textContent   = '0.00ms';
    if (rVal)         rVal.textContent    = '0.000';
    if (ecoVal)       ecoVal.textContent  = '0%';
    if (ecoBar)       ecoBar.style.width  = '0%';
    if (ecoHopVal) { ecoHopVal.textContent = '+0.00 dB'; ecoHopVal.style.color = 'var(--accent-cyan)'; }
    if (predictDisp)  predictDisp.textContent = '0%';
    if (predictBar)   predictBar.style.width   = '0%';
    if (predConfVal)  predConfVal.textContent  = '---';
    if (priorityDisp) priorityDisp.textContent = 'NONE';
    const plsVal2 = document.getElementById('pls-val');
    const plsBar  = document.getElementById('pls-bar');
    if (plsVal2) { plsVal2.textContent = '100%'; plsVal2.style.color = '#10b981'; }
    if (plsBar)  { plsBar.style.width = '100%'; plsBar.style.background = '#10b981'; }
    // Clear log
    if (rLog) rLog.innerHTML = '';
    // Reset charts
    if (lossChart) { lossChart.data.datasets[0].data = new Array(20).fill(0); lossChart.update('none'); }
    learningSteps = 0;
    if (stepVal) stepVal.textContent = '0';
    logToTerminal('HUD reset by user.', 'success');
});

// AI Settings Sliders
aiThreshSlider?.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if(aiThreshDisp) aiThreshDisp.textContent = val.toFixed(2);
    ws.send(JSON.stringify({ cmd: 'update_params', ai_thresh: val }));
});
histWSlider?.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if(histWDisp) histWDisp.textContent = val.toFixed(2);
    ws.send(JSON.stringify({ cmd: 'update_params', hist_w: val }));
});
uncertWSlider?.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if(uncertWDisp) uncertWDisp.textContent = val.toFixed(2);
    ws.send(JSON.stringify({ cmd: 'update_params', uncert_w: val }));
});

// Network Slice Selector
document.getElementById('slice-select')?.addEventListener('change', (e) => {
    const val = e.target.value;
    const prioVal = document.getElementById('priority-val');
    
    // Immediate visual update to match Sci-Fi aesthetic
    if (prioVal) {
        if (val === 'urllc') {
            prioVal.textContent = 'CRITICAL LATENCY';
            prioVal.style.color = '#ef4444';
        } else if (val === 'embb') {
            prioVal.textContent = 'HIGH BANDWIDTH';
            prioVal.style.color = '#22d3ee';
        } else if (val === 'mmtc') {
            prioVal.textContent = 'ECO SENSOR NODE';
            prioVal.style.color = '#10b981';
        } else {
            prioVal.textContent = 'NONE';
            prioVal.style.color = '#cbd5e1';
        }
    }
    
    // Dispatch to Python Engine
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ cmd: 'set_network_slice', val: val }));
    }
});

// Tab Switching logic
const tabLoss = document.getElementById('tab-loss');
const tabAcc = document.getElementById('tab-acc');
const lossContainer = document.getElementById('loss-chart-container');
const accContainer = document.getElementById('acc-chart-container');

tabLoss?.addEventListener('click', () => {
    tabLoss.classList.add('active');
    tabAcc.classList.remove('active');
    if (lossContainer) lossContainer.style.display = 'block';
    if (accContainer) accContainer.style.display = 'none';
});

tabAcc?.addEventListener('click', () => {
    tabAcc.classList.add('active');
    tabLoss.classList.remove('active');
    if (accContainer) accContainer.style.display = 'block';
    if (lossContainer) lossContainer.style.display = 'none';
});

// Initialize Chart.js
function initCharts() {
    const accCanvas = document.getElementById('acc-canvas');
    if (!accCanvas || typeof Chart === 'undefined') return;
    
    // Fill initial 0s
    predictionHistory = new Array(CHART_MAX_POINTS).fill(0);
    actualHistory = new Array(CHART_MAX_POINTS).fill(0);
    const labels = new Array(CHART_MAX_POINTS).fill('');
    
    accChart = new Chart(accCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Probability',
                    data: predictionHistory,
                    borderColor: '#22d3ee',
                    backgroundColor: 'rgba(34, 211, 238, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Actual Channel State',
                    data: actualHistory,
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    stepped: true,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { size: 9, family: '"JetBrains Mono", monospace' } }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 1.1,
                    ticks: { color: '#64748b', font: { size: 9 } },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                x: {
                    display: false
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // initRadar, initFreqMatrix, initCharts are called here;
    // initFrequencyGrid, initLossChart, initMIMO are called in the 'load' event above.
    initCharts();
});
