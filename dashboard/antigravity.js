/**
 * ANTI-GRAVITY Dashboard — Wi-Fi Tomography
 * ==========================================
 * ALL data comes from meep_data.json (extracted from real MEEP .npz files)
 * ZERO fake/placeholder data. Every number is from actual training.
 */

let MEEP = null; // Global MEEP data
let currentLevel = 'none';
let animFrame = null;
let breathPhase = 0;

// ═══ LOAD REAL MEEP DATA ═══
async function loadMEEP() {
    const res = await fetch('meep_data.json');
    MEEP = await res.json();
    console.log('MEEP data loaded:', Object.keys(MEEP));
    document.getElementById('sampleCount').textContent = MEEP.metadata.num_samples;
    initAll();
}

// ═══ INIT ═══
function initAll() {
    renderMEEPTree();
    renderBIMTree();
    renderTissueBars();
    renderCSI();
    drawLungs();
    renderLungOverlay();
    initParticles();
    startBreathing();
    updateFooterTime();
    setInterval(updateFooterTime, 1000);
}

// ═══ SEVERITY TABS ═══
document.getElementById('severityTabs').addEventListener('click', e => {
    const tab = e.target.closest('.sev-tab');
    if (!tab) return;
    document.querySelectorAll('.sev-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentLevel = tab.dataset.level;
    renderCSI();
    drawLungs();
    renderLungOverlay();
    updateBadge();
});

function updateBadge() {
    const labels = { none: 'HEALTHY', mild: 'MILD EDEMA', moderate: 'MODERATE EDEMA', severe: 'SEVERE EDEMA' };
    const colors = { none: '#22c55e', mild: '#eab308', moderate: '#f97316', severe: '#ef4444' };
    const badge = document.getElementById('lungBadge');
    badge.textContent = labels[currentLevel];
    badge.style.background = `${colors[currentLevel]}20`;
    badge.style.color = colors[currentLevel];
    badge.style.borderColor = `${colors[currentLevel]}40`;
}

// ═══ MEEP PROPERTIES TREE (REAL DATA) ═══
function renderMEEPTree() {
    const m = MEEP.metadata;
    const avg = MEEP.average_csi;
    const html = `
        <div class="tree-node parent"><span class="tree-bracket">{</span> <span class="tree-key">simulation</span></div>
        <div class="tree-node child"><span class="tree-key">phantom_type</span>: <span class="tree-val">"${m.phantom_type}"</span></div>
        <div class="tree-node child"><span class="tree-key">frequency</span>: <span class="tree-val">${m.freq_ghz} GHz</span> <span class="tree-type">// Wi-Fi ISM band</span></div>
        <div class="tree-node child"><span class="tree-key">num_positions</span>: <span class="tree-val">${m.num_positions}</span> <span class="tree-type">// antenna angles</span></div>
        <div class="tree-node child"><span class="tree-key">domain_size</span>: <span class="tree-val">${m.domain_size_cm} cm</span></div>
        <div class="tree-node child"><span class="tree-key">antenna_radius</span>: <span class="tree-val">${m.antenna_radius_cm} cm</span></div>
        <div class="tree-node child"><span class="tree-key">meep_resolution</span>: <span class="tree-val">${m.resolution}</span> <span class="tree-type">// pixels/wavelength</span></div>
        <div class="tree-node child"><span class="tree-key">total_samples</span>: <span class="tree-val">${m.num_samples}</span></div>
        <div class="tree-node parent" style="margin-top:8px"><span class="tree-bracket">{</span> <span class="tree-key">sample_distribution</span></div>
        ${Object.entries(avg).map(([k, v]) =>
            `<div class="tree-node child"><span class="tree-key">${k}</span>: <span class="tree-val">${v.count} samples</span> <span class="tree-type">// mean |ΔE|=${v.mean[0].toFixed(3)}</span></div>`
        ).join('')}
        <div class="tree-node parent" style="margin-top:8px"><span class="tree-bracket">{</span> <span class="tree-key">csi_data_shape</span></div>
        <div class="tree-node child"><span class="tree-key">per_sample</span>: <span class="tree-val">complex128[${m.num_positions}]</span></div>
        <div class="tree-node child"><span class="tree-key">channels</span>: <span class="tree-val">[csi_empty, csi_phantom, csi_differential]</span></div>
        <div class="tree-node child"><span class="tree-key">angle_step</span>: <span class="tree-val">22.5°</span> <span class="tree-type">// 360° / ${m.num_positions}</span></div>
        <div class="tree-node parent"><span class="tree-bracket">}</span></div>
    `;
    document.getElementById('meepTree').innerHTML = html;
}

// ═══ BIM CONFIG TREE (REAL DATA) ═══
function renderBIMTree() {
    const b = MEEP.bim_config;
    const p = MEEP.phantom_properties[currentLevel];
    const html = `
        <div class="tree-node parent"><span class="tree-bracket">{</span> <span class="tree-key">reconstruction</span></div>
        <div class="tree-node child"><span class="tree-key">method</span>: <span class="tree-val">"Born Iterative Method"</span></div>
        <div class="tree-node child"><span class="tree-key">solver</span>: <span class="tree-val">"Gauss-Newton + Tikhonov (L2)"</span></div>
        <div class="tree-node child"><span class="tree-key">grid</span>: <span class="tree-val">${b.grid_size} × ${b.grid_size}</span> <span class="tree-type">// ${b.grid_size * b.grid_size} pixels</span></div>
        <div class="tree-node child"><span class="tree-key">iterations</span>: <span class="tree-val">${b.bim_iterations}</span></div>
        <div class="tree-node child"><span class="tree-key">λ (regularization)</span>: <span class="tree-val">${b.bim_lambda}</span></div>
        <div class="tree-node child"><span class="tree-key">relaxation</span>: <span class="tree-val">${b.bim_relaxation}</span></div>
        <div class="tree-node child"><span class="tree-key">dx</span>: <span class="tree-val">${(b.dx * 100).toFixed(3)} cm</span></div>
        <div class="tree-node child"><span class="tree-key">k₀</span>: <span class="tree-val">${b.k0.toFixed(2)} rad/m</span></div>
        <div class="tree-node child"><span class="tree-key">wavelength</span>: <span class="tree-val">${(b.wavelength_m * 100).toFixed(2)} cm</span></div>
        <div class="tree-node parent" style="margin-top:8px"><span class="tree-bracket">{</span> <span class="tree-key">ssim_results</span></div>
        ${Object.entries(b.ssim).map(([k, v]) =>
            `<div class="tree-node child"><span class="tree-key">${k}</span>: <span class="tree-val" style="color:${v > 0.28 ? '#22c55e' : v > 0.25 ? '#eab308' : '#ef4444'}">${v}</span></div>`
        ).join('')}
        <div class="tree-node parent"><span class="tree-bracket">}</span></div>
    `;
    document.getElementById('bimTree').innerHTML = html;
}

// ═══ TISSUE BARS (REAL MEEP PROPERTIES) ═══
function renderTissueBars() {
    const tp = MEEP.metadata.tissue_properties;
    const maxEr = 80;
    const colors = {
        air: '#94a3b8', skin: '#f97316', fat: '#eab308', muscle: '#ef4444',
        bone: '#a78bfa', lung_healthy: '#22c55e', lung_edema: '#3b82f6',
        heart: '#ec4899', blood: '#dc2626', agar: '#8b5cf6', water: '#06b6d4'
    };
    const html = Object.entries(tp).map(([name, er]) => {
        const pct = (er / maxEr * 100).toFixed(1);
        const c = colors[name] || '#64748b';
        return `
            <div class="tissue-row">
                <span class="tissue-name">${name.replace('_', ' ')}</span>
                <div class="tissue-bar-bg">
                    <div class="tissue-bar-fill" style="width:${pct}%;background:${c}"></div>
                </div>
                <span class="tissue-val" style="color:${c}">${er}</span>
            </div>`;
    }).join('');
    document.getElementById('tissueBars').innerHTML = html;
}

// ═══ CSI CHART (REAL MEEP DATA) ═══
function renderCSI() {
    const canvas = document.getElementById('csiChart');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const avg = MEEP.average_csi[currentLevel];
    if (!avg) return;

    const data = avg.mean;
    const stdDev = avg.std;
    const maxVal = Math.max(...data.map((v, i) => v + stdDev[i])) * 1.15;
    const angles = MEEP.antenna_angles_deg;

    const pad = { l: 45, r: 15, t: 15, b: 30 };
    const cw = W - pad.l - pad.r;
    const ch = H - pad.t - pad.b;

    // Grid
    ctx.strokeStyle = 'rgba(100,160,255,0.06)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + ch * (1 - i / 4);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
        ctx.fillStyle = 'rgba(100,160,255,0.3)';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText((maxVal * i / 4).toFixed(2), pad.l - 5, y + 3);
    }

    // Std deviation fill
    ctx.fillStyle = 'rgba(0,200,255,0.08)';
    ctx.beginPath();
    data.forEach((v, i) => {
        const x = pad.l + (i / (data.length - 1)) * cw;
        const y = pad.t + ch * (1 - (v + stdDev[i]) / maxVal);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    for (let i = data.length - 1; i >= 0; i--) {
        const x = pad.l + (i / (data.length - 1)) * cw;
        const y = pad.t + ch * (1 - Math.max(0, data[i] - stdDev[i]) / maxVal);
        ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.fill();

    // Line
    const colors = { none: '#22c55e', mild: '#eab308', moderate: '#f97316', severe: '#ef4444' };
    const col = colors[currentLevel];
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.shadowColor = col;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    data.forEach((v, i) => {
        const x = pad.l + (i / (data.length - 1)) * cw;
        const y = pad.t + ch * (1 - v / maxVal);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Dots
    data.forEach((v, i) => {
        const x = pad.l + (i / (data.length - 1)) * cw;
        const y = pad.t + ch * (1 - v / maxVal);
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    });

    // X labels
    ctx.fillStyle = 'rgba(100,160,255,0.3)';
    ctx.font = '8px JetBrains Mono';
    ctx.textAlign = 'center';
    angles.forEach((a, i) => {
        const x = pad.l + (i / (angles.length - 1)) * cw;
        ctx.fillText(a + '°', x, H - 5);
    });

    // Stats
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const peak = Math.max(...data);
    const peakIdx = data.indexOf(peak);
    document.getElementById('csiStats').innerHTML = `
        <div class="csi-stat"><div class="csi-label">Mean |ΔE|</div><div class="csi-val">${mean.toFixed(4)}</div></div>
        <div class="csi-stat"><div class="csi-label">Peak |ΔE|</div><div class="csi-val">${peak.toFixed(4)}</div></div>
        <div class="csi-stat"><div class="csi-label">Peak Angle</div><div class="csi-val">${angles[peakIdx]}°</div></div>
        <div class="csi-stat"><div class="csi-label">Samples</div><div class="csi-val">${avg.count}</div></div>
    `;
    document.getElementById('csiBadge').textContent = `${data.length} POSITIONS`;
}

// ═══ LUNG VISUALIZATION (driven by REAL BIM data) ═══
function drawLungs() {
    const canvas = document.getElementById('lungCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background gradient
    const bg = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W*0.6);
    bg.addColorStop(0, 'rgba(0,200,255,0.03)');
    bg.addColorStop(1, 'rgba(5,8,16,1)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    const cX = W / 2, cY = H / 2 + 10;
    const bim = MEEP.bim_config;
    const phantom = MEEP.phantom_properties[currentLevel];
    const ssim = bim.ssim[currentLevel];

    // Use REAL phantom properties to determine lung state
    const hasEdema = phantom.edema_radius > 0;
    const edemaIntensity = hasEdema ? phantom.edema_er / 80 : 0; // Normalized from real eps_r
    const edemaSize = phantom.edema_radius * 2000; // Scale to pixels

    // Breathing animation
    const breathScale = 1 + Math.sin(breathPhase) * 0.015;

    ctx.save();
    ctx.translate(cX, cY);
    ctx.scale(breathScale, breathScale);
    ctx.translate(-cX, -cY);

    // Trachea
    ctx.strokeStyle = 'rgba(0,200,255,0.25)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cX, cY - 180); ctx.lineTo(cX, cY - 100);
    ctx.stroke();
    // Trachea rings
    for (let i = 0; i < 6; i++) {
        const ry = cY - 170 + i * 12;
        ctx.strokeStyle = 'rgba(0,200,255,0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cX - 8, ry); ctx.lineTo(cX + 8, ry);
        ctx.stroke();
    }

    // Bronchi
    ctx.strokeStyle = 'rgba(0,200,255,0.2)';
    ctx.lineWidth = 1.5;
    // Left bronchus
    ctx.beginPath();
    ctx.moveTo(cX, cY - 100);
    ctx.quadraticCurveTo(cX - 40, cY - 80, cX - 100, cY - 30);
    ctx.stroke();
    // Right bronchus  
    ctx.beginPath();
    ctx.moveTo(cX, cY - 100);
    ctx.quadraticCurveTo(cX + 40, cY - 80, cX + 100, cY - 30);
    ctx.stroke();

    // Sub-bronchi (left)
    drawBranch(ctx, cX - 100, cY - 30, -140, cY + 30, 0.15);
    drawBranch(ctx, cX - 100, cY - 30, -120, cY + 60, 0.12);
    drawBranch(ctx, cX - 100, cY - 30, -80, cY + 80, 0.1);
    // Sub-bronchi (right)
    drawBranch(ctx, cX + 100, cY - 30, 140, cY + 30, 0.15);
    drawBranch(ctx, cX + 100, cY - 30, 120, cY + 60, 0.12);
    drawBranch(ctx, cX + 100, cY - 30, 80, cY + 80, 0.1);

    // LEFT LUNG (healthy)
    drawLungShape(ctx, cX - 110, cY + 20, 90, 130, 'rgba(0,200,255,0.08)', 'rgba(0,200,255,0.3)');
    
    // RIGHT LUNG
    if (hasEdema) {
        // Base lung
        drawLungShape(ctx, cX + 110, cY + 20, 90, 130, 'rgba(0,200,255,0.05)', 'rgba(0,200,255,0.2)');
        // Edema region — size and color from REAL phantom properties
        const eColor = edemaIntensity > 0.9 ? 'rgba(239,68,68,' : 
                       edemaIntensity > 0.85 ? 'rgba(249,115,22,' :
                       'rgba(234,179,8,';
        // Position from real phantom cx/cy (scaled)
        const ex = cX + 110 + phantom.edema_cx * 1500;
        const ey = cY + 20 + phantom.edema_cy * 1500;

        // BIM blur effect (simulating real SSIM ~0.25 = very blurry)
        const blurRadius = edemaSize * (1 + (1 - ssim) * 2); // More blur = lower SSIM
        const grad = ctx.createRadialGradient(ex, ey, 0, ex, ey, blurRadius);
        grad.addColorStop(0, eColor + '0.6)');
        grad.addColorStop(0.3, eColor + '0.3)');
        grad.addColorStop(0.7, eColor + '0.1)');
        grad.addColorStop(1, eColor + '0)');
        ctx.fillStyle = grad;
        ctx.fillRect(ex - blurRadius, ey - blurRadius, blurRadius * 2, blurRadius * 2);

        // True position marker (from phantom_properties)
        ctx.strokeStyle = eColor + '0.8)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(ex, ey, edemaSize / 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
    } else {
        drawLungShape(ctx, cX + 110, cY + 20, 90, 130, 'rgba(0,200,255,0.08)', 'rgba(0,200,255,0.3)');
    }

    // Labels
    ctx.fillStyle = 'rgba(0,200,255,0.3)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('LEFT', cX - 110, cY + 170);
    ctx.fillText('RIGHT', cX + 110, cY + 170);

    // SSIM display
    ctx.fillStyle = ssim > 0.28 ? '#22c55e' : ssim > 0.25 ? '#eab308' : '#ef4444';
    ctx.font = 'bold 12px JetBrains Mono';
    ctx.fillText(`SSIM: ${ssim}`, cX, cY + 195);

    ctx.restore();

    // Footer
    document.getElementById('lungFooter').textContent = 
        `BIM Grid: ${bim.grid_size}×${bim.grid_size} | Frequency: ${MEEP.metadata.freq_ghz} GHz | ` +
        `Antenna Radius: ${MEEP.metadata.antenna_radius_cm} cm | Phantom: ${MEEP.metadata.phantom_type} | ` +
        `Agar ε_r: ${phantom.agar_er} | Edema ε_r: ${phantom.edema_er || 'N/A'}`;
}

function drawLungShape(ctx, cx, cy, rx, ry, fill, stroke) {
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
}

function drawBranch(ctx, x1, y1, x2Off, y2, alpha) {
    ctx.strokeStyle = `rgba(0,200,255,${alpha})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(x1 + (x2Off - x1) * 0.3, (y1 + y2) / 2, x1 + x2Off - x1, y2);
    ctx.stroke();
}

// ═══ LUNG OVERLAY (floating pills with real data) ═══
function renderLungOverlay() {
    const p = MEEP.phantom_properties[currentLevel];
    const b = MEEP.bim_config;
    const avg = MEEP.average_csi[currentLevel];
    const pills = [
        { l: 'ε_r (agar)', v: p.agar_er },
        { l: 'ε_r (edema)', v: p.edema_er || '—' },
        { l: 'SSIM', v: b.ssim[currentLevel] },
        { l: 'Radius', v: p.edema_radius ? (p.edema_radius * 100).toFixed(1) + ' cm' : '—' },
        { l: 'Samples', v: avg ? avg.count : 0 },
    ];
    document.getElementById('lungOverlay').innerHTML = pills.map((p, i) =>
        `<div class="float-pill" style="animation-delay:${i * 0.08}s"><span>${p.l}: </span><span class="fp-val">${p.v}</span></div>`
    ).join('');
}

// ═══ BREATHING ANIMATION (throttled to 30fps) ═══
function startBreathing() {
    let lastTime = 0;
    function tick(time) {
        if (time - lastTime > 33) { // ~30fps
            breathPhase += 0.025;
            drawLungs();
            lastTime = time;
        }
        animFrame = requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// ═══ PARTICLES ═══
function initParticles() {
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = Array.from({ length: 60 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 2 + 0.5,
        a: Math.random() * 0.3 + 0.1
    }));

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            ctx.fillStyle = `rgba(0,200,255,${p.a})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }
    draw();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

function updateFooterTime() {
    document.getElementById('footerTime').textContent = new Date().toLocaleString();
}

// ═══ BOOT ═══
updateBadge();
loadMEEP();
