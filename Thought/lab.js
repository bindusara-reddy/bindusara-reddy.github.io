/* ============================================================
   a real neural network, living in this page
   engine + animated forward-pass visualization + teaching loop
   ============================================================ */

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const VIO = '#8b7cff', VIO2 = '#b9adff', AMBER = '#ffc466', PINK = '#ff6f9c',
      INK = '#eceae4', MUT = '#8e8c99';

/* ---------- the network engine (784 -> 64 -> 10) ---------- */

const NN = (() => {
    const IN = 784, HID = MNIST_WEIGHTS.hid, OUT = 10;
    const STORE_KEY = 'nn-digit-weights-v1';
    const COUNT_KEY = 'nn-digit-taught-v1';
    let w1, b1, w2, b2;

    function b64ToFloats(s, scale) {
        const bin = atob(s);
        const out = new Float32Array(bin.length);
        for (let i = 0; i < bin.length; i++) {
            let v = bin.charCodeAt(i);
            if (v > 127) v -= 256;
            out[i] = v * scale;
        }
        return out;
    }

    function factory() {
        w1 = b64ToFloats(MNIST_WEIGHTS.w1, MNIST_WEIGHTS.s1);
        b1 = Float32Array.from(MNIST_WEIGHTS.b1);
        w2 = b64ToFloats(MNIST_WEIGHTS.w2, MNIST_WEIGHTS.s2);
        b2 = Float32Array.from(MNIST_WEIGHTS.b2);
    }

    function serialize() {
        const all = new Float32Array(w1.length + b1.length + w2.length + b2.length);
        all.set(w1, 0);
        all.set(b1, w1.length);
        all.set(w2, w1.length + b1.length);
        all.set(b2, w1.length + b1.length + w2.length);
        const bytes = new Uint8Array(all.buffer);
        let bin = '';
        const CH = 8192;
        for (let i = 0; i < bytes.length; i += CH) {
            bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
        }
        return btoa(bin);
    }

    function save() {
        try { localStorage.setItem(STORE_KEY, serialize()); } catch (e) {}
    }

    function restore() {
        try {
            const s = localStorage.getItem(STORE_KEY);
            if (!s) return false;
            const bin = atob(s);
            if (bin.length !== (w1.length + b1.length + w2.length + b2.length) * 4) return false;
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            const all = new Float32Array(bytes.buffer);
            let o = 0;
            w1 = all.slice(o, o += w1.length);
            b1 = all.slice(o, o += b1.length);
            w2 = all.slice(o, o += w2.length);
            b2 = all.slice(o, o += b2.length);
            return true;
        } catch (e) { return false; }
    }

    function reset() {
        factory();
        try {
            localStorage.removeItem(STORE_KEY);
            localStorage.removeItem(COUNT_KEY);
        } catch (e) {}
    }

    function forward(x) {
        const a1 = new Float32Array(HID);
        for (let h = 0; h < HID; h++) {
            let s = b1[h];
            const off = h * IN;
            for (let i = 0; i < IN; i++) s += w1[off + i] * x[i];
            a1[h] = s > 0 ? s : 0;
        }
        const z2 = new Float32Array(OUT);
        let mx = -Infinity;
        for (let o = 0; o < OUT; o++) {
            let s = b2[o];
            const off = o * HID;
            for (let h = 0; h < HID; h++) s += w2[off + h] * a1[h];
            z2[o] = s;
            if (s > mx) mx = s;
        }
        let sum = 0;
        const probs = new Float32Array(OUT);
        for (let o = 0; o < OUT; o++) {
            probs[o] = Math.exp(z2[o] - mx);
            sum += probs[o];
        }
        for (let o = 0; o < OUT; o++) probs[o] /= sum;
        return { a1, probs };
    }

    function trainStep(x, label, lr = 0.012, reps = 4) {
        let loss = 0;
        for (let r = 0; r < reps; r++) {
            const { a1, probs } = forward(x);
            loss = -Math.log(Math.max(probs[label], 1e-9));
            const dz2 = Float32Array.from(probs);
            dz2[label] -= 1;
            const dz1 = new Float32Array(HID);
            for (let o = 0; o < OUT; o++) {
                const off = o * HID;
                for (let h = 0; h < HID; h++) dz1[h] += w2[off + h] * dz2[o];
            }
            for (let h = 0; h < HID; h++) if (a1[h] <= 0) dz1[h] = 0;
            for (let o = 0; o < OUT; o++) {
                const off = o * HID;
                const g = dz2[o] * lr;
                for (let h = 0; h < HID; h++) w2[off + h] -= g * a1[h];
                b2[o] -= g;
            }
            for (let h = 0; h < HID; h++) {
                if (dz1[h] === 0) continue;
                const off = h * IN;
                const g = dz1[h] * lr;
                for (let i = 0; i < IN; i++) w1[off + i] -= g * x[i];
                b1[h] -= g;
            }
        }
        save();
        return loss;
    }

    const taughtCount = () => parseInt(localStorage.getItem(COUNT_KEY) || '0', 10);
    function bumpTaught() {
        const n = taughtCount() + 1;
        try { localStorage.setItem(COUNT_KEY, String(n)); } catch (e) {}
        return n;
    }
    function getW2() { return w2; }

    factory();
    const restored = restore();
    return { forward, trainStep, reset, taughtCount, bumpTaught, restored, getW1: () => w1, getW2, HID };
})();

/* ---------- scroll progress + chapter reveals ---------- */

(() => {
    const bar = document.getElementById('progress');
    addEventListener('scroll', () => {
        const max = document.documentElement.scrollHeight - innerHeight;
        bar.style.width = (max > 0 ? (scrollY / max) * 100 : 0) + '%';
    }, { passive: true });

    if (!REDUCED && 'IntersectionObserver' in window) {
        const io = new IntersectionObserver(es => es.forEach(e => {
            if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }
        }), { threshold: 0.1 });
        document.querySelectorAll('.reveal').forEach(el => io.observe(el));
    } else {
        document.querySelectorAll('.reveal').forEach(el => el.classList.add('visible'));
    }
})();

/* ---------- hero: a drifting constellation network ---------- */

(() => {
    const cv = document.getElementById('hero-net');
    const ctx = cv.getContext('2d');
    let W, H, nodes = [], pulses = [], running = true, frame = 0;

    function fit() {
        W = cv.clientWidth; H = cv.clientHeight;
        cv.width = W; cv.height = H;
    }

    function init() {
        fit();
        const n = Math.min(70, Math.floor(W * H / 16000));
        nodes = Array.from({ length: n }, () => ({
            x: Math.random() * W, y: Math.random() * H,
            vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25
        }));
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        const R = 130;
        for (let i = 0; i < nodes.length; i++) {
            const a = nodes[i];
            for (let j = i + 1; j < nodes.length; j++) {
                const b = nodes[j];
                const dx = a.x - b.x, dy = a.y - b.y;
                const d2 = dx * dx + dy * dy;
                if (d2 < R * R) {
                    const t = 1 - Math.sqrt(d2) / R;
                    ctx.strokeStyle = `rgba(139,124,255,${(t * 0.16).toFixed(3)})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }
        for (const n of nodes) {
            ctx.fillStyle = 'rgba(185,173,255,0.5)';
            ctx.beginPath();
            ctx.arc(n.x, n.y, 1.6, 0, Math.PI * 2);
            ctx.fill();
        }
        for (const p of pulses) {
            const t = p.t;
            const x = p.a.x + (p.b.x - p.a.x) * t;
            const y = p.a.y + (p.b.y - p.a.y) * t;
            ctx.fillStyle = `rgba(255,196,102,${(1 - t) * 0.9})`;
            ctx.beginPath();
            ctx.arc(x, y, 2.6, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function tick() {
        if (!running) return;
        frame++;
        if (frame % 2 === 0) {
            for (const n of nodes) {
                n.x += n.vx; n.y += n.vy;
                if (n.x < 0 || n.x > W) n.vx *= -1;
                if (n.y < 0 || n.y > H) n.vy *= -1;
            }
            for (const p of pulses) p.t += 0.03;
            pulses = pulses.filter(p => p.t < 1);
            if (Math.random() < 0.05 && pulses.length < 6 && nodes.length > 1) {
                const a = nodes[(Math.random() * nodes.length) | 0];
                let best = null, bd = 1e9;
                for (const b of nodes) {
                    if (b === a) continue;
                    const d = (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
                    if (d < bd && d < 130 * 130) { bd = d; best = b; }
                }
                if (best) pulses.push({ a, b: best, t: 0 });
            }
            draw();
        }
        requestAnimationFrame(tick);
    }

    init();
    if (REDUCED) { draw(); return; }
    addEventListener('resize', init);
    document.addEventListener('visibilitychange', () => {
        running = !document.hidden;
        if (running) tick();
    });
    new IntersectionObserver(es => {
        const vis = es[0].isIntersecting;
        if (vis && !running) { running = true; tick(); }
        if (!vis) running = false;
    }).observe(cv);
    tick();
})();

/* ---------- chapter 01: the neuron playground ---------- */

(() => {
    const ids = ['nx1', 'nw1', 'nx2', 'nw2', 'nb'];
    const sig = z => 1 / (1 + Math.exp(-z));

    function update() {
        const [x1, w1, x2, w2, b] = ids.map(i => parseFloat(document.getElementById(i).value));
        const z = x1 * w1 + x2 * w2 + b;
        const a = sig(z);
        ids.forEach(i => {
            document.getElementById(i + '-v').textContent =
                parseFloat(document.getElementById(i).value).toFixed(2);
        });
        const l1 = document.getElementById('nl1');
        const l2 = document.getElementById('nl2');
        l1.setAttribute('stroke-width', (0.6 + Math.abs(w1) * 3.2).toFixed(2));
        l2.setAttribute('stroke-width', (0.6 + Math.abs(w2) * 3.2).toFixed(2));
        l1.setAttribute('stroke', w1 >= 0 ? VIO : PINK);
        l2.setAttribute('stroke', w2 >= 0 ? VIO : PINK);
        l1.setAttribute('opacity', (0.35 + Math.min(1, Math.abs(w1) / 2) * 0.65).toFixed(2));
        l2.setAttribute('opacity', (0.35 + Math.min(1, Math.abs(w2) / 2) * 0.65).toFixed(2));
        const lo = document.getElementById('nlo');
        lo.setAttribute('stroke-width', (0.5 + a * 5).toFixed(2));
        lo.setAttribute('opacity', (0.25 + a * 0.75).toFixed(2));
        document.getElementById('ncell').setAttribute('fill', `rgba(139,124,255,${(0.1 + 0.45 * a).toFixed(2)})`);
        const out = document.getElementById('nout');
        out.setAttribute('r', (10 + a * 8).toFixed(1));
        out.setAttribute('fill', `rgba(255,196,102,${(0.12 + 0.7 * a).toFixed(2)})`);
        document.getElementById('nx1-t').textContent = x1.toFixed(1);
        document.getElementById('nx2-t').textContent = x2.toFixed(1);
        document.getElementById('nz-t').textContent = z.toFixed(2);
        document.getElementById('na-t').textContent = a.toFixed(2);
        document.getElementById('neuron-formula').textContent =
            `${x1.toFixed(2)}×${w1.toFixed(2)} + ${x2.toFixed(2)}×${w2.toFixed(2)} + ${b.toFixed(2)} = ${z.toFixed(2)}  →  squashed to ${a.toFixed(2)}`;
    }

    ids.forEach(i => document.getElementById(i).addEventListener('input', update));
    update();
})();

/* ---------- chapter 02: what the hidden neurons learned ---------- */

(() => {
    const w1 = NN.getW1();
    const norms = [];
    for (let h = 0; h < NN.HID; h++) {
        let s = 0;
        for (let i = 0; i < 784; i++) s += w1[h * 784 + i] ** 2;
        norms.push([s, h]);
    }
    norms.sort((a, b) => b[0] - a[0]);
    const host = document.getElementById('features');
    for (let k = 0; k < 8; k++) {
        const h = norms[k][1];
        let mx = 0;
        for (let i = 0; i < 784; i++) mx = Math.max(mx, Math.abs(w1[h * 784 + i]));
        const c = document.createElement('canvas');
        c.width = 28; c.height = 28;
        c.className = 'feature';
        c.title = `hidden neuron #${h}`;
        const img = c.getContext('2d').createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const v = w1[h * 784 + i] / mx;
            const o = i * 4;
            if (v >= 0) {
                img.data[o] = Math.round(139 * v + 7);
                img.data[o + 1] = Math.round(124 * v + 7);
                img.data[o + 2] = Math.round(255 * v + 11);
            } else {
                img.data[o] = Math.round(255 * -v + 7);
                img.data[o + 1] = Math.round(111 * -v + 7);
                img.data[o + 2] = Math.round(156 * -v + 11);
            }
            img.data[o + 3] = 255;
        }
        c.getContext('2d').putImageData(img, 0, 0);
        host.appendChild(c);
    }
})();

/* ---------- chapter 03: gradient descent with a trail ---------- */

(() => {
    const cv = document.getElementById('bowl');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const TARGET = 1.1;
    let w = -2.7, trail = [], autoTimer = null;
    const loss = v => (v - TARGET) * (v - TARGET);
    const X = v => (v + 3.6) / 7.2 * W;
    const Y = v => H - 36 - loss(v) / loss(-3.6) * (H - 80);

    function draw() {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = 'rgba(139,124,255,0.9)';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let px = 0; px <= W; px += 6) {
            const v = px / W * 7.2 - 3.6;
            px === 0 ? ctx.moveTo(px, Y(v)) : ctx.lineTo(px, Y(v));
        }
        ctx.stroke();
        ctx.fillStyle = 'rgba(142,140,153,0.8)';
        ctx.font = '15px JetBrains Mono, monospace';
        ctx.fillText('loss', 16, 28);
        ctx.fillText('weight →', W - 110, H - 12);
        trail.forEach((v, i) => {
            const t = (i + 1) / trail.length;
            ctx.fillStyle = `rgba(255,196,102,${(t * 0.35).toFixed(2)})`;
            ctx.beginPath();
            ctx.arc(X(v), Y(v), 5, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.beginPath();
        ctx.arc(X(w), Y(w), 11, 0, Math.PI * 2);
        ctx.fillStyle = AMBER;
        ctx.shadowColor = 'rgba(255,196,102,0.9)';
        ctx.shadowBlur = 26;
        ctx.fill();
        ctx.shadowBlur = 0;
        document.getElementById('bowl-stats').textContent =
            `w = ${w.toFixed(2)}  loss = ${loss(w).toFixed(3)}`;
    }

    function step() {
        trail.push(w);
        if (trail.length > 26) trail.shift();
        const lr = parseFloat(document.getElementById('lr').value);
        w -= lr * 2 * (w - TARGET);
        w = Math.max(-3.5, Math.min(3.5, w));
        document.getElementById('lr-warn').textContent =
            lr > 0.9 ? '⚠ stride too long — it keeps leaping over the valley' : '';
        draw();
    }

    document.getElementById('step-btn').addEventListener('click', step);
    document.getElementById('auto-btn').addEventListener('click', function () {
        if (autoTimer) {
            clearInterval(autoTimer);
            autoTimer = null;
            this.textContent = 'auto-descend';
        } else {
            autoTimer = setInterval(step, 260);
            this.textContent = 'stop';
        }
    });
    document.getElementById('drop-btn').addEventListener('click', () => {
        w = -2.7 + Math.random() * 0.9;
        trail = [];
        draw();
    });
    document.getElementById('lr').addEventListener('input', () => {
        document.getElementById('lr-v').textContent =
            parseFloat(document.getElementById('lr').value).toFixed(2);
    });
    draw();
})();

/* ---------- chapter 04: the lab ---------- */

const Lab = (() => {
    const pad = document.getElementById('pad');
    const pctx = pad.getContext('2d');
    const viz = document.getElementById('netviz');
    const vctx = viz.getContext('2d');
    const VW = 660, VH = 420;
    viz.width = VW * 2; viz.height = VH * 2;
    vctx.scale(2, 2);

    let drawing = false, lastPt = null, hasInk = false;
    let state = null;          // { x, a1, probs, winner }
    let animId = null;
    const losses = [];

    /* ----- drawing pad ----- */

    function clearPadPixels() {
        pctx.fillStyle = '#0d0d14';
        pctx.fillRect(0, 0, pad.width, pad.height);
    }

    function clearPad() {
        clearPadPixels();
        hasInk = false;
        state = null;
        const v = document.getElementById('verdict');
        v.textContent = 'draw a digit…';
        v.classList.add('idle');
        document.getElementById('confidence').textContent = '';
        document.getElementById('teach-row').classList.remove('armed');
        renderViz(1);
    }

    function pos(e) {
        const r = pad.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * pad.width / r.width,
            y: (e.clientY - r.top) * pad.height / r.height
        };
    }

    pad.addEventListener('pointerdown', e => {
        e.preventDefault();
        pad.setPointerCapture(e.pointerId);
        drawing = true;
        lastPt = pos(e);
    });
    pad.addEventListener('pointermove', e => {
        if (!drawing) return;
        const p = pos(e);
        pctx.strokeStyle = '#fff';
        pctx.lineWidth = 22;
        pctx.lineCap = 'round';
        pctx.lineJoin = 'round';
        pctx.beginPath();
        pctx.moveTo(lastPt.x, lastPt.y);
        pctx.lineTo(p.x, p.y);
        pctx.stroke();
        lastPt = p;
        hasInk = true;
    });
    const stopDraw = () => {
        if (drawing) {
            drawing = false;
            if (hasInk) predict();
        }
    };
    pad.addEventListener('pointerup', stopDraw);
    pad.addEventListener('pointercancel', stopDraw);

    /* ----- MNIST-style preprocessing (crop, scale to 20, recentre by mass) ----- */

    function preprocess() {
        const W = pad.width, H = pad.height;
        const d = pctx.getImageData(0, 0, W, H).data;
        let minX = W, minY = H, maxX = -1, maxY = -1;
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                if (d[(y * W + x) * 4] > 40) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return null;
        const boxW = maxX - minX + 1, boxH = maxY - minY + 1;
        const scale = 20 / Math.max(boxW, boxH);
        const sw = Math.max(1, Math.round(boxW * scale));
        const sh = Math.max(1, Math.round(boxH * scale));
        const tmp = document.createElement('canvas');
        tmp.width = 28; tmp.height = 28;
        const tctx = tmp.getContext('2d');
        tctx.fillStyle = '#000';
        tctx.fillRect(0, 0, 28, 28);
        tctx.drawImage(pad, minX, minY, boxW, boxH,
            Math.floor((28 - sw) / 2), Math.floor((28 - sh) / 2), sw, sh);
        const td = tctx.getImageData(0, 0, 28, 28).data;
        let m = 0, mx = 0, my = 0;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const v = td[(y * 28 + x) * 4];
                m += v; mx += x * v; my += y * v;
            }
        }
        const dx = Math.round(13.5 - mx / m), dy = Math.round(13.5 - my / m);
        const out = document.createElement('canvas');
        out.width = 28; out.height = 28;
        const octx = out.getContext('2d');
        octx.fillStyle = '#000';
        octx.fillRect(0, 0, 28, 28);
        octx.drawImage(tmp, dx, dy);
        const od = octx.getImageData(0, 0, 28, 28).data;
        const x784 = new Float32Array(784);
        for (let i = 0; i < 784; i++) x784[i] = od[i * 4] / 255;
        return x784;
    }

    /* ----- the animated forward-pass diagram ----- */

    // layout (logical 660x420)
    const THUMB = { x: 16, y: 162, s: 96 };
    const GRID = { x: 178, y: 152, cols: 8, rows: 8, gap: 16.5, r: 4.5 };
    const OUTS = { x: 380, y0: 30, dy: 38.5, node: 7, label: 402, bar0: 424, bar1: 588, pct: 596 };

    const easeOut = t => 1 - Math.pow(1 - t, 3);
    const clamp01 = t => Math.max(0, Math.min(1, t));

    function dotPos(h) {
        return {
            x: GRID.x + (h % GRID.cols) * GRID.gap,
            y: GRID.y + Math.floor(h / GRID.cols) * GRID.gap
        };
    }

    function drawThumb(x784, alpha) {
        vctx.save();
        vctx.globalAlpha = alpha;
        vctx.strokeStyle = 'rgba(139,124,255,0.4)';
        vctx.lineWidth = 1;
        vctx.strokeRect(THUMB.x, THUMB.y, THUMB.s, THUMB.s);
        if (x784) {
            const cell = THUMB.s / 28;
            for (let i = 0; i < 784; i++) {
                const v = x784[i];
                if (v < 0.04) continue;
                vctx.fillStyle = `rgba(185,173,255,${(v).toFixed(2)})`;
                vctx.fillRect(THUMB.x + (i % 28) * cell, THUMB.y + Math.floor(i / 28) * cell, cell + 0.5, cell + 0.5);
            }
        } else {
            vctx.fillStyle = 'rgba(142,140,153,0.5)';
            vctx.font = '10px JetBrains Mono, monospace';
            vctx.fillText('28×28', THUMB.x + 30, THUMB.y + THUMB.s / 2 + 3);
        }
        vctx.fillStyle = 'rgba(142,140,153,0.75)';
        vctx.font = '10px JetBrains Mono, monospace';
        vctx.fillText('your digit', THUMB.x + 14, THUMB.y + THUMB.s + 18);
        vctx.fillText('784 inputs', THUMB.x + 14, THUMB.y + THUMB.s + 32);
        vctx.restore();
    }

    function renderViz(t) {
        const s = state;
        vctx.clearRect(0, 0, VW, VH);

        const t1 = easeOut(clamp01(t / 0.4));          // input -> hidden
        const t2 = easeOut(clamp01((t - 0.28) / 0.4)); // hidden -> output
        const t3 = easeOut(clamp01((t - 0.5) / 0.5));  // bars

        drawThumb(s ? s.x : null, 1);

        // normalise activations
        let maxA = 0;
        if (s) for (let h = 0; h < NN.HID; h++) maxA = Math.max(maxA, s.a1[h]);

        // input -> hidden wires
        if (s && maxA > 0) {
            const sx = THUMB.x + THUMB.s, sy = THUMB.y + THUMB.s / 2;
            for (let h = 0; h < NN.HID; h++) {
                const act = s.a1[h] / maxA;
                if (act < 0.05) continue;
                const p = dotPos(h);
                const ex = sx + (p.x - sx) * t1, ey = sy + (p.y - sy) * t1;
                vctx.strokeStyle = `rgba(139,124,255,${(act * 0.34).toFixed(3)})`;
                vctx.lineWidth = 0.8;
                vctx.beginPath();
                vctx.moveTo(sx, sy);
                vctx.lineTo(ex, ey);
                vctx.stroke();
            }
        }

        // hidden dots
        for (let h = 0; h < NN.HID; h++) {
            const p = dotPos(h);
            const act = s && maxA > 0 ? s.a1[h] / maxA : 0;
            const pop = s ? t1 : 1;
            const r = GRID.r * (0.7 + 0.5 * act * pop);
            vctx.beginPath();
            vctx.arc(p.x, p.y, r, 0, Math.PI * 2);
            vctx.fillStyle = `rgba(185,173,255,${(0.12 + 0.88 * act * pop).toFixed(3)})`;
            vctx.fill();
        }
        vctx.fillStyle = 'rgba(142,140,153,0.75)';
        vctx.font = '10px JetBrains Mono, monospace';
        vctx.fillText('64 hidden neurons', GRID.x - 4, GRID.y + 7 * GRID.gap + 26);

        // hidden -> output wires (top contributions per output)
        if (s && t2 > 0) {
            const w2 = NN.getW2();
            for (let o = 0; o < 10; o++) {
                const contribs = [];
                for (let h = 0; h < NN.HID; h++) {
                    const c = s.a1[h] * w2[o * NN.HID + h];
                    if (c !== 0) contribs.push([Math.abs(c), c, h]);
                }
                contribs.sort((a, b) => b[0] - a[0]);
                const top = contribs.slice(0, 6);
                const maxC = top.length ? top[0][0] : 1;
                const oy = OUTS.y0 + o * OUTS.dy;
                for (const [ac, c, h] of top) {
                    const p = dotPos(h);
                    const ex = p.x + (OUTS.x - p.x) * t2, ey = p.y + (oy - p.y) * t2;
                    const k = ac / maxC;
                    const base = o === s.winner ? 0.55 : 0.10;
                    vctx.strokeStyle = c >= 0
                        ? (o === s.winner ? `rgba(255,196,102,${(base * k).toFixed(3)})` : `rgba(139,124,255,${(base * k).toFixed(3)})`)
                        : `rgba(255,111,156,${(base * k * 0.9).toFixed(3)})`;
                    vctx.lineWidth = o === s.winner ? 1.3 : 0.8;
                    vctx.beginPath();
                    vctx.moveTo(p.x, p.y);
                    vctx.lineTo(ex, ey);
                    vctx.stroke();
                }
            }
        }

        // output nodes + bars
        for (let o = 0; o < 10; o++) {
            const oy = OUTS.y0 + o * OUTS.dy;
            const prob = s ? s.probs[o] : 0;
            const win = s && o === s.winner;
            const lit = s ? prob * t3 : 0;

            vctx.beginPath();
            vctx.arc(OUTS.x, oy, OUTS.node, 0, Math.PI * 2);
            vctx.fillStyle = win
                ? `rgba(255,196,102,${(0.2 + 0.8 * t3).toFixed(2)})`
                : `rgba(185,173,255,${(0.1 + 0.55 * lit).toFixed(2)})`;
            vctx.fill();

            vctx.font = (win ? '700 ' : '') + '13px JetBrains Mono, monospace';
            vctx.fillStyle = win ? AMBER : 'rgba(201,199,210,0.8)';
            vctx.fillText(String(o), OUTS.label, oy + 4.5);

            vctx.fillStyle = 'rgba(255,255,255,0.06)';
            roundBar(OUTS.bar0, oy - 5.5, OUTS.bar1 - OUTS.bar0, 11);
            if (lit > 0.002) {
                vctx.fillStyle = win ? AMBER : 'rgba(139,124,255,0.75)';
                roundBar(OUTS.bar0, oy - 5.5, Math.max(3, (OUTS.bar1 - OUTS.bar0) * lit), 11);
            }
            vctx.font = '11px JetBrains Mono, monospace';
            vctx.fillStyle = win ? 'rgba(255,196,102,0.95)' : 'rgba(142,140,153,0.85)';
            vctx.fillText(Math.round((s ? prob : 0) * t3 * 100) + '%', OUTS.pct, oy + 4);
        }
    }

    function roundBar(x, y, w, h) {
        const r = h / 2;
        vctx.beginPath();
        vctx.moveTo(x + r, y);
        vctx.arcTo(x + w, y, x + w, y + h, r);
        vctx.arcTo(x + w, y + h, x, y + h, r);
        vctx.arcTo(x, y + h, x, y, r);
        vctx.arcTo(x, y, x + w, y, r);
        vctx.fill();
    }

    function animate() {
        if (animId) cancelAnimationFrame(animId);
        if (REDUCED) { renderViz(1); return; }
        const t0 = performance.now();
        const DUR = 850;
        const loop = now => {
            const t = clamp01((now - t0) / DUR);
            renderViz(t);
            if (t < 1) animId = requestAnimationFrame(loop);
        };
        animId = requestAnimationFrame(loop);
    }

    /* ----- predict + teach ----- */

    function predict() {
        const x = preprocess();
        if (!x) return;
        const { a1, probs } = NN.forward(x);
        let winner = 0;
        for (let d = 1; d < 10; d++) if (probs[d] > probs[winner]) winner = d;
        state = { x, a1, probs, winner };
        const v = document.getElementById('verdict');
        v.textContent = String(winner);
        v.classList.remove('idle');
        const conf = probs[winner];
        document.getElementById('confidence').textContent =
            conf > 0.85 ? `${Math.round(conf * 100)}% sure` :
            conf > 0.5 ? `${Math.round(conf * 100)}% sure… probably` :
            `${Math.round(conf * 100)}% sure — teach me?`;
        document.getElementById('teach-row').classList.add('armed');
        animate();
        return winner;
    }

    function drawSpark() {
        const cv = document.getElementById('loss-spark');
        const c = cv.getContext('2d');
        const W = cv.width, H = cv.height;
        c.clearRect(0, 0, W, H);
        if (losses.length < 1) return;
        const data = losses.slice(-30);
        const mx = Math.max(...data, 0.5);
        c.strokeStyle = 'rgba(255,255,255,0.12)';
        c.beginPath(); c.moveTo(0, H - 8); c.lineTo(W, H - 8); c.stroke();
        c.strokeStyle = AMBER;
        c.lineWidth = 2.5;
        c.beginPath();
        data.forEach((l, i) => {
            const x = data.length === 1 ? W / 2 : 8 + i / (data.length - 1) * (W - 16);
            const y = H - 8 - (l / mx) * (H - 20);
            i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
        });
        c.stroke();
        data.forEach((l, i) => {
            const x = data.length === 1 ? W / 2 : 8 + i / (data.length - 1) * (W - 16);
            const y = H - 8 - (l / mx) * (H - 20);
            c.beginPath(); c.arc(x, y, 3.5, 0, Math.PI * 2);
            c.fillStyle = AMBER; c.fill();
        });
    }

    function teach(label) {
        if (!state) return;
        const loss = NN.trainStep(state.x, label);
        const n = NN.bumpTaught();
        document.getElementById('taught-count').textContent = n;
        losses.push(loss);
        document.getElementById('spark-wrap').classList.add('show');
        drawSpark();
        document.getElementById('teach-note').textContent =
            `learned. loss ${loss.toFixed(3)} — watch the wires shift →`;
        predict();
    }

    document.getElementById('clear-btn').addEventListener('click', clearPad);
    document.querySelectorAll('#teach-row button[data-digit]').forEach(btn => {
        btn.addEventListener('click', () => teach(parseInt(btn.dataset.digit, 10)));
    });
    document.getElementById('reset-btn').addEventListener('click', () => {
        if (confirm('Forget everything you taught it and restore the factory brain?')) {
            NN.reset();
            document.getElementById('taught-count').textContent = '0';
            document.getElementById('teach-note').textContent = '';
            losses.length = 0;
            document.getElementById('spark-wrap').classList.remove('show');
            if (state) predict();
        }
    });

    document.getElementById('taught-count').textContent = NN.taughtCount();
    if (NN.restored) {
        document.getElementById('restored-note').textContent = 'this browser remembers your lessons';
    }
    clearPadPixels();
    clearPad();
    return { predict, teach, clearPad };
})();
