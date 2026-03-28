const ws = new WebSocket(`ws://${window.location.host}/ws`);

// DOM Elements
const latVal = document.getElementById('latency-val');
const chVal = document.getElementById('channel-val');
const rVal = document.getElementById('reward-val'); // Used for D3QN Mock Loss display
const hoVal = document.getElementById('handover-val');
const pwrVal = document.getElementById('power-val');
const stepVal = document.getElementById('steps-val');

// System state tracking
let learningSteps = parseInt(stepVal?.textContent) || 0;

// Dynamic Frequency Tabs Setup
const freqTabsContainer = document.querySelector('.frequency-tabs');
if (freqTabsContainer) {
    freqTabsContainer.innerHTML = '';
    const displayBands = ['2.40G', '2.42G', '2.44G', '2.46G', '2.48G']; // Map 10 channels to 5 UI tabs
    displayBands.forEach((band, idx) => {
        const btn = document.createElement('button');
        btn.className = idx === 0 ? 'tab active' : 'tab';
        btn.textContent = band;
        freqTabsContainer.appendChild(btn);
    });
}

function updateTabs(chIdx) {
    const visualIdx = Math.floor((chIdx || 0) / 2); // 0-9 index divided by 2 into 0-4 tabs
    const tabs = document.querySelectorAll('.frequency-tabs .tab');
    if(tabs.length > visualIdx) {
        tabs.forEach(t => t.classList.remove('active'));
        tabs[visualIdx].classList.add('active');
        
        // Update Frequency metric text
        if(chVal) chVal.textContent = `${tabs[visualIdx].textContent}Hz`;
    }
}

// Waterfall Canvas
const canvas = document.getElementById('waterfall-canvas');
const ctx = canvas.getContext('2d');
let waterfallHistory = [];
const HISTORY_LEN = 80;

function resizeCanvas() {
    if(canvas && canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
    }
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

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
            // Render beautiful sleek neon heatmaps 
            const normalized = Math.max(0, Math.min(1, (val + 80) / 100)); 
            if(normalized < 0.1) continue; // Skip rendering dark for performance and aesthetic
            
            // HSL map to match the cyberpunk/cyan theme
            const hue = 190 + (normalized * 90); 
            ctx.fillStyle = `hsla(${hue}, 90%, 60%, ${normalized * 0.9})`;
            ctx.fillRect(j * cellW, y, Math.ceil(cellW)+1, Math.ceil(cellH)+1);
        }
    }
}

function updateMockEvoChart() {
    // Generate pseudo-data for the pink "D3QN Loss" chart to make it jitter realistically
    if(window.lossChart && window.lossChart.data.datasets.length > 0) {
        let arr = window.lossChart.data.datasets[0].data;
        arr.shift();
        let targetLoss = Math.max(10, 50 - (learningSteps / 20)); // Base trend: loss dropping then oscillating
        let newLoss = targetLoss + (Math.random() * 20 - 10);
        arr.push(Math.max(0, newLoss));
        window.lossChart.update();
        if(rVal) rVal.textContent = newLoss.toFixed(3);
    }
}

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Check SDR Link Status Overlay
    if (data.sdr_linked === false) {
        document.getElementById('freeze-overlay').style.display = 'flex';
        const modeTag = document.querySelector('.mode-tag.linked');
        if(modeTag) {
            modeTag.textContent = 'SDR: ERROR';
            modeTag.style.color = '#ef4444';
        }
        return; // Freeze updates
    } else {
        const overlay = document.getElementById('freeze-overlay');
        if (overlay) overlay.style.display = 'none';
        const modeTag = document.querySelector('.mode-tag.linked');
        if(modeTag) {
            modeTag.textContent = 'SDR: LINKED';
            modeTag.style.color = '#14b8a6';
        }
    }
    
    // Base Telemetry bindings
    if(latVal) {
        latVal.textContent = `${data.latency_ms} ms`;
        latVal.style.color = data.latency_ms > 30 ? '#fcd34d' : '#38bdf8';
    }
    
    updateTabs(data.channel_idx);

    // AI logic parsing
    if (data.is_busy) {
        // We had a forced handover step
        learningSteps++;
        if(stepVal) stepVal.textContent = learningSteps;
        if(hoVal) hoVal.textContent = `${data.latency_ms} ms`;
        updateMockEvoChart();
    } else {
        // Channel IDLE
        if(hoVal) hoVal.textContent = `0.00ms`;
    }
    
    // Draw UI PSD Slice & Set Power Stat
    drawWaterfall(data.waterfall_slice);
    if(data.waterfall_slice && data.waterfall_slice.length > 0) {
        const maxPwr = Math.max(...data.waterfall_slice);
        if(pwrVal) pwrVal.textContent = `${maxPwr.toFixed(1)} dBm`;
    }
    
    // Update Frequency Matrix Chart
    if (data.q_values && data.q_values.length > 0 && freqChart) {
        freqChart.data.datasets[0].data = data.q_values;
        const colors = new Array(10).fill('rgba(56, 189, 248, 0.4)');
        colors[data.channel_idx] = 'rgba(20, 184, 166, 0.8)'; // Teal for active
        if (data.is_busy) colors[data.channel_idx] = 'rgba(239, 68, 68, 0.8)'; // Red
        freqChart.data.datasets[0].backgroundColor = colors;
        freqChart.update();
    }
};

// Hidden logic bindings to prevent old js from throwing warnings if imported later
document.getElementById('twin-toggle')?.addEventListener('change', (e) => {
    ws.send(JSON.stringify({ cmd: 'toggle_twin', val: e.target.checked }));
});
document.getElementById('throughput-opt')?.addEventListener('change', (e) => {
    ws.send(JSON.stringify({ cmd: 'toggle_optimizer', val: e.target.checked }));
});
document.getElementById('exit-btn')?.addEventListener('click', () => {
    ws.send(JSON.stringify({ cmd: 'exit' }));
});

// Frequency Matrix Chart Initialization
const freqMatrixCtx = document.getElementById('freq-matrix-canvas')?.getContext('2d');
const freqLabels = Array.from({length: 10}, (_, i) => `${2400 + (i * 10)} MHz`);
const freqChart = freqMatrixCtx ? new Chart(freqMatrixCtx, {
    type: 'bar',
    data: {
        labels: freqLabels,
        datasets: [{
            label: 'AI Model Q-Value (Channel Quality)',
            data: new Array(10).fill(0),
            backgroundColor: 'rgba(56, 189, 248, 0.4)',
            borderColor: '#38bdf8',
            borderWidth: 1,
            borderRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#9ba6b5', font: {family: 'JetBrains Mono', size: 10} } },
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ba6b5', font: {size: 10} } }
        },
        animation: { duration: 100 }
    }
}) : null;

// View More Button Toggle Event
document.getElementById('view-more-btn')?.addEventListener('click', function() {
    this.classList.toggle('open');
    const panel = document.getElementById('expanded-freq-panel');
    if (panel) panel.classList.toggle('open');
});
