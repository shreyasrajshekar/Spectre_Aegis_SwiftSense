// --- Internal Config & State ---
// Default labels match Legacy ISM band (433 MHz → ~5.8 GHz), synced from backend on connect
let freqLabels = ['433M', '1.0G', '1.6G', '2.2G', '2.4G', '2.8G', '3.4G', '4.0G', '5.2G', '5.8G'];
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
const pwrDisp = document.getElementById('pwr-val');
const predictDisp = document.getElementById('predict-val');
const predictBar = document.getElementById('predict-bar');
const priorityDisp = document.getElementById('priority-val');
const ecoVal = document.getElementById('eco-val');
const ecoBar = document.getElementById('eco-bar');
const plsVal = document.getElementById('pls-val');
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
});

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
        predictBar.style.width = pStr;
        predictBar.style.backgroundColor = data.prediction_horizon > 0.8 ? '#ef4444' : '#22d3ee';
    }

    drawMIMO(data.active_beam);
    drawRadar(data.radar_map);

    if(ecoVal && ecoBar) {
        const ecoStr = data.eco_saving + '%';
        ecoVal.textContent = ecoStr;
        ecoBar.style.width = ecoStr;
    }
    
    if(plsVal) {
        plsVal.textContent = data.pls_score + '%';
        plsVal.style.color = data.pls_score > 80 ? '#10b981' : data.pls_score > 40 ? '#fbbf24' : '#ef4444';
    }

    if(confVal) confVal.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    
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
