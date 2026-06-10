/* A real neural network living in your browser.
   Loads weights trained on MNIST, predicts digits you draw,
   and fine-tunes itself on every digit you teach it. */

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
            const expected = (w1.length + b1.length + w2.length + b2.length) * 4;
            if (bin.length !== expected) return false;
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

    function taughtCount() {
        return parseInt(localStorage.getItem(COUNT_KEY) || '0', 10);
    }

    function bumpTaught() {
        const n = taughtCount() + 1;
        try { localStorage.setItem(COUNT_KEY, String(n)); } catch (e) {}
        return n;
    }

    factory();
    const restored = restore();
    return { forward, trainStep, reset, taughtCount, bumpTaught, restored, getW1: () => w1, HID };
})();

/* ---------- drawing canvas + MNIST-style preprocessing ---------- */

const Lab = (() => {
    const pad = document.getElementById('pad');
    const pctx = pad.getContext('2d');
    let drawing = false, lastPt = null, hasInk = false, lastX = null;

    function clearPad() {
        pctx.fillStyle = '#000';
        pctx.fillRect(0, 0, pad.width, pad.height);
        hasInk = false;
        lastX = null;
        document.getElementById('verdict').textContent = 'draw a digit';
        document.getElementById('confidence').textContent = '';
        document.getElementById('teach-row').classList.remove('armed');
        renderBars(new Float32Array(10), -1);
        renderHidden(new Float32Array(NN.HID));
        const seen = document.getElementById('seen');
        seen.getContext('2d').clearRect(0, 0, seen.width, seen.height);
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
        pctx.lineWidth = 20;
        pctx.lineCap = 'round';
        pctx.lineJoin = 'round';
        pctx.beginPath();
        pctx.moveTo(lastPt.x, lastPt.y);
        pctx.lineTo(p.x, p.y);
        pctx.stroke();
        lastPt = p;
        hasInk = true;
    });
    const stop = () => {
        if (drawing) {
            drawing = false;
            if (hasInk) predict();
        }
    };
    pad.addEventListener('pointerup', stop);
    pad.addEventListener('pointercancel', stop);

    // crop the ink, scale it to 20px, recentre by centre of mass into 28x28 —
    // the same recipe the MNIST dataset itself was made with
    function preprocess() {
        const W = pad.width, H = pad.height;
        const d = pctx.getImageData(0, 0, W, H).data;
        let minX = W, minY = H, maxX = -1, maxY = -1;
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                if (d[(y * W + x) * 4] > 20) {
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
        tctx.imageSmoothingEnabled = true;
        tctx.drawImage(pad, minX, minY, boxW, boxH, Math.floor((28 - sw) / 2), Math.floor((28 - sh) / 2), sw, sh);

        let td = tctx.getImageData(0, 0, 28, 28).data;
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

    function renderSeen(x) {
        const seen = document.getElementById('seen');
        const sctx = seen.getContext('2d');
        const img = sctx.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const v = Math.round(x[i] * 255);
            img.data[i * 4] = Math.round(v * 0.34);
            img.data[i * 4 + 1] = Math.round(v * 0.92);
            img.data[i * 4 + 2] = Math.round(v * 0.96);
            img.data[i * 4 + 3] = 255;
        }
        const tiny = document.createElement('canvas');
        tiny.width = 28; tiny.height = 28;
        tiny.getContext('2d').putImageData(img, 0, 0);
        sctx.imageSmoothingEnabled = false;
        sctx.clearRect(0, 0, seen.width, seen.height);
        sctx.drawImage(tiny, 0, 0, seen.width, seen.height);
    }

    function renderHidden(a1) {
        const cells = document.querySelectorAll('#hidden-grid i');
        let mx = 0;
        for (let h = 0; h < a1.length; h++) if (a1[h] > mx) mx = a1[h];
        cells.forEach((c, h) => {
            const t = mx > 0 ? a1[h] / mx : 0;
            c.style.opacity = (0.08 + 0.92 * t).toFixed(2);
        });
    }

    function renderBars(probs, winner) {
        for (let d = 0; d < 10; d++) {
            const row = document.getElementById('bar-' + d);
            const fill = row.querySelector('b');
            const pct = row.querySelector('em');
            fill.style.width = (probs[d] * 100).toFixed(1) + '%';
            pct.textContent = (probs[d] * 100).toFixed(0) + '%';
            row.classList.toggle('winner', d === winner);
        }
    }

    function predict() {
        const x = preprocess();
        if (!x) return;
        lastX = x;
        renderSeen(x);
        const { a1, probs } = NN.forward(x);
        let best = 0;
        for (let d = 1; d < 10; d++) if (probs[d] > probs[best]) best = d;
        renderHidden(a1);
        renderBars(probs, best);
        const conf = probs[best];
        document.getElementById('verdict').textContent = String(best);
        document.getElementById('confidence').textContent =
            conf > 0.85 ? `${(conf * 100).toFixed(0)}% sure` :
            conf > 0.5 ? `${(conf * 100).toFixed(0)}% sure… probably` :
            `only ${(conf * 100).toFixed(0)}% sure — teach me?`;
        document.getElementById('teach-row').classList.add('armed');
        return best;
    }

    function teach(label) {
        if (!lastX) return;
        const loss = NN.trainStep(lastX, label);
        const n = NN.bumpTaught();
        document.getElementById('taught-count').textContent = n;
        const note = document.getElementById('teach-note');
        note.textContent = `learned it (loss ${loss.toFixed(3)} → lower is better) — watch the bars shift:`;
        predict();
        const row = document.getElementById('bar-' + label);
        row.classList.remove('flash');
        void row.offsetWidth;
        row.classList.add('flash');
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
            if (lastX) predict();
        }
    });

    const grid = document.getElementById('hidden-grid');
    for (let h = 0; h < NN.HID; h++) grid.appendChild(document.createElement('i'));
    document.getElementById('taught-count').textContent = NN.taughtCount();
    clearPad();
    return { predict, teach, clearPad };
})();

/* ---------- section 1: a single neuron you can poke ---------- */

(() => {
    const ids = ['nx1', 'nw1', 'nx2', 'nw2', 'nb'];
    function update() {
        const [x1, w1, x2, w2, b] = ids.map(i => parseFloat(document.getElementById(i).value));
        const z = x1 * w1 + x2 * w2 + b;
        const a = 1 / (1 + Math.exp(-z));
        document.getElementById('neuron-formula').textContent =
            `${x1.toFixed(1)}×${w1.toFixed(1)} + ${x2.toFixed(1)}×${w2.toFixed(1)} + ${b.toFixed(1)} = ${z.toFixed(2)}`;
        document.getElementById('neuron-out').textContent = a.toFixed(2);
        const orb = document.getElementById('neuron-orb');
        orb.style.boxShadow = `0 0 ${8 + 60 * a}px rgba(87,234,245,${(0.15 + 0.85 * a).toFixed(2)})`;
        orb.style.background = `rgba(87,234,245,${(0.08 + 0.72 * a).toFixed(2)})`;
        ids.forEach(i => {
            document.getElementById(i + '-v').textContent = parseFloat(document.getElementById(i).value).toFixed(1);
        });
    }
    ids.forEach(i => document.getElementById(i).addEventListener('input', update));
    update();
})();

/* ---------- section 2: what the hidden neurons actually learned ---------- */

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
        const img = c.getContext('2d').createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const v = w1[h * 784 + i] / mx;
            const o = i * 4;
            if (v >= 0) {
                img.data[o] = Math.round(87 * v);
                img.data[o + 1] = Math.round(234 * v);
                img.data[o + 2] = Math.round(245 * v);
            } else {
                img.data[o] = Math.round(232 * -v);
                img.data[o + 1] = Math.round(78 * -v);
                img.data[o + 2] = Math.round(166 * -v);
            }
            img.data[o + 3] = 255;
        }
        c.getContext('2d').putImageData(img, 0, 0);
        host.appendChild(c);
    }
})();

/* ---------- section 3: gradient descent on a bowl ---------- */

(() => {
    const cv = document.getElementById('bowl');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const TARGET = 0.9;
    let w = -2.4, autoTimer = null;
    const loss = v => (v - TARGET) * (v - TARGET);
    const X = v => (v + 3.2) / 6.4 * W;
    const Y = v => H - 24 - loss(v) / loss(-3.2) * (H - 60);

    function draw() {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = 'rgba(87,234,245,0.85)';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let px = 0; px <= W; px += 4) {
            const v = px / W * 6.4 - 3.2;
            px === 0 ? ctx.moveTo(px, Y(v)) : ctx.lineTo(px, Y(v));
        }
        ctx.stroke();
        ctx.fillStyle = 'rgba(161,255,235,0.55)';
        ctx.font = '13px Roboto Mono, monospace';
        ctx.fillText('loss', 10, 20);
        ctx.fillText('weight →', W - 80, H - 6);
        ctx.beginPath();
        ctx.arc(X(w), Y(w), 9, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.shadowColor = 'rgba(87,234,245,1)';
        ctx.shadowBlur = 18;
        ctx.fill();
        ctx.shadowBlur = 0;
        document.getElementById('bowl-stats').textContent =
            `weight = ${w.toFixed(2)}   loss = ${loss(w).toFixed(3)}`;
    }

    function step() {
        const lr = parseFloat(document.getElementById('lr').value);
        w -= lr * 2 * (w - TARGET);
        w = Math.max(-3.1, Math.min(3.1, w));
        draw();
    }

    document.getElementById('step-btn').addEventListener('click', step);
    document.getElementById('auto-btn').addEventListener('click', function () {
        if (autoTimer) {
            clearInterval(autoTimer);
            autoTimer = null;
            this.textContent = 'auto-descend';
        } else {
            autoTimer = setInterval(step, 280);
            this.textContent = 'stop';
        }
    });
    document.getElementById('drop-btn').addEventListener('click', () => {
        w = -2.4 + Math.random() * 0.8;
        draw();
    });
    document.getElementById('lr').addEventListener('input', () => {
        document.getElementById('lr-v').textContent = document.getElementById('lr').value;
    });
    draw();
})();

if (NN.restored) {
    const el = document.getElementById('restored-note');
    if (el) el.textContent = 'this browser remembered your training from last time';
}
