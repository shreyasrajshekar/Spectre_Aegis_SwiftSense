// --- Internal Config & State ---
const freqLabels = Array.from({length: 10}, (_, i) => `${(2.4 + (i * 0.01)).toFixed(2)}G`);
let currentViewMode = 'spectrum'; 
let waterfallHistory = [];
const HISTORY_LEN = 80;
let freqChart = null;
let lossChart = null;
let lastChartUpdate = 0;

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
const viewToggleBtn = document.getElementById('view-toggle-btn');
const viewModeIcon = document.getElementById('view-mode-icon');
const canvas = document.getElementById('spectrum-canvas');
const ctx = canvas?.getContext('2d');
const alertCollision = document.getElementById('alert-collision');
const alertHandover = document.getElementById('alert-handover');

// --- Initialization ---
let learningSteps = parseInt(stepVal?.textContent) || 0;

function setupTabs() {
    const container = document.querySelector('.frequency-tabs');
    if (!container) return;
    container.innerHTML = '';
    
    // Display first 5 bands for the HUD tabs
    freqLabels.slice(0, 5).forEach((label, idx) => {
        const btn = document.createElement('button');
        btn.className = 'tab';
        btn.textContent = label;
        btn.onclick = () => {
            if(window.ws) window.ws.send(JSON.stringify({ cmd: 'set_channel', idx: idx * 2 }));
            updateTabUI(idx);
        };
        container.appendChild(btn);
    });
}

function initFreqMatrix() {
    const freqMatrixCtx = document.getElementById('freq-matrix-canvas')?.getContext('2d');
    if(freqMatrixCtx) {
        freqChart = new Chart(freqMatrixCtx, {
            type: 'bar',
            data: { labels: freqLabels, datasets: [{ data: new Array(10).fill(0), borderRadius: 4, backgroundColor: 'rgba(56, 189, 248, 0.3)' }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: false },
                scales: {
                    x: { ticks: { color: '#94a3b8', font: {size: 9} } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { display: false } }
                }
            }
        });
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
    const tabs = document.querySelectorAll('.frequency-tabs .tab');
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
});
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
    const timestamp = new Date().toLocaleTimeString('en-GB', { hour12: false, fractionDigits: 1 });
    line.innerHTML = `<span style="color:var(--text-muted)">[${timestamp}]</span> ${msg}`;
    rLog.appendChild(line);
    rLog.scrollTop = rLog.scrollHeight;
    if(rLog.childNodes.length > 100) rLog.removeChild(rLog.firstChild);
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
    if(chVal) chVal.textContent = freqLabels[data.channel_idx] + "Hz";
    if(rVal) rVal.textContent = data.reward.toFixed(3);
    if(pwrVal && data.waterfall_slice.length > 0) {
        const peak = Math.max(...data.waterfall_slice);
        pwrVal.textContent = `${peak.toFixed(1)} dBm`;
    }
    
    // Update AI Engine Graph (Loss/Reward Chart)
    if(lossChart) {
        const val = (data.reward !== undefined && !isNaN(data.reward)) ? data.reward : 0;
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
    
    if (currentViewMode === 'spectrum') drawSpectrum(data.waterfall_slice);
    else drawWaterfall(data.waterfall_slice);
    
    if (data.q_values && data.q_values.length > 0 && freqChart) {
        freqChart.data.datasets[0].data = data.q_values;
        const colors = new Array(10).fill('rgba(56, 189, 248, 0.3)');
        colors[data.channel_idx] = data.is_busy ? 'rgba(239, 68, 68, 0.8)' : 'rgba(20, 184, 166, 0.8)';
        freqChart.data.datasets[0].backgroundColor = colors;
        freqChart.update('none');
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
    currentViewMode = currentViewMode === 'spectrum' ? 'waterfall' : 'spectrum';
    if(viewModeIcon) viewModeIcon.textContent = currentViewMode === 'spectrum' ? '📊' : '🌊';
    waterfallHistory = [];
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
