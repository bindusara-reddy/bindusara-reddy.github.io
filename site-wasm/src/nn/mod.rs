//! The "how a machine learns to read" page: a six-chapter explorable
//! explanation wrapped around the real network in engine.rs.

mod bowl;
mod features;
mod hero;
mod lab;
mod neuron;
mod numbers;

use crate::util::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

pub fn render() {
    let net = crate::engine::Net::factory();
    let body = body_html(net.acc);
    document().body().unwrap().set_inner_html(&body);

    progress_bar();
    reveals();
    hero::init();
    numbers::init();
    neuron::init();
    features::init();
    bowl::init();
    lab::init();
}

fn progress_bar() {
    on(window().as_ref(), "scroll", |_| {
        let doc = document().document_element().unwrap();
        let max = doc.scroll_height() as f64 - window().inner_height().unwrap().as_f64().unwrap();
        let y = window().scroll_y().unwrap_or(0.0);
        let w = if max > 0.0 { y / max * 100.0 } else { 0.0 };
        let _ = get_html("progress").style().set_property("width", &format!("{w}%"));
    });
}

fn reveals() {
    let all = document().query_selector_all(".reveal").unwrap();
    if reduced_motion() {
        for i in 0..all.length() {
            let el: web_sys::Element = all.item(i).unwrap().dyn_into().unwrap();
            let _ = el.class_list().add_1("visible");
        }
        return;
    }
    let cb = Closure::wrap(Box::new(
        move |entries: js_sys::Array, obs: web_sys::IntersectionObserver| {
            for e in entries.iter() {
                let entry: web_sys::IntersectionObserverEntry = e.dyn_into().unwrap();
                if entry.is_intersecting() {
                    let el = entry.target();
                    let _ = el.class_list().add_1("visible");
                    obs.unobserve(&el);
                }
            }
        },
    )
        as Box<dyn FnMut(js_sys::Array, web_sys::IntersectionObserver)>);
    if let Ok(obs) = web_sys::IntersectionObserver::new(cb.as_ref().unchecked_ref()) {
        for i in 0..all.length() {
            let el: web_sys::Element = all.item(i).unwrap().dyn_into().unwrap();
            obs.observe(&el);
        }
    }
    cb.forget();
}

fn body_html(acc: f32) -> String {
    let acc = format!("{acc:.1}");
    let head = format!(
        r##"<div id="progress"></div>
    <a class="home" href="../index.html">&larr; home</a>
    <main>
        <section class="hero">
            <canvas id="hero-net" aria-hidden="true"></canvas>
            <div class="hero-inner">
                <p class="kicker">a living explainer</p>
                <h1>how a machine<br><em>learns to read</em></h1>
                <p class="lede">There is a real neural network woven into this page — 50,890 numbers
                    that taught themselves to read handwriting. Scroll down to take it apart,
                    then retrain it with your own hands.</p>
                <div class="chips">
                    <span class="chip">784 → 64 → 10 neurons</span>
                    <span class="chip">{acc}% accurate</span>
                    <span class="chip">compiled from rust to wasm</span>
                    <span class="chip">learns from you</span>
                </div>
            </div>
            <div class="scroll-cue" aria-hidden="true"></div>
        </section>"##
    );

    let chapters = r##"
        <section class="chapter reveal">
            <span class="chapnum">chapter 01</span>
            <h2>it sees numbers, not pictures</h2>
            <p>Computers don't see. When you draw a digit, your sketch is melted down to a
                28-by-28 grid of brightness values — <span class="mono">784</span> numbers,
                each between 0 and 1. Below is a seven, the way the network receives it:
                <strong>just the numbers</strong> (0 = black ink missing, 9 = brightest ink).</p>
            <div class="card">
                <canvas id="numbers" width="1232" height="1232"></canvas>
                <p class="legend">this is the entire input — there is no picture, only this grid</p>
            </div>
            <p style="margin-top:22px">Everything that follows is arithmetic on that list of numbers.
                Hold on to how strange that is: <em>arithmetic alone is about to read.</em></p>
        </section>

        <section class="chapter reveal">
            <span class="chapnum">chapter 02</span>
            <h2>a neuron is a tiny opinion</h2>
            <p>One neuron does embarrassingly little. It looks at some inputs and weighs them up like
                evidence. Each <strong>weight</strong> says how much one input matters — positive means
                <em>"bright here? that's my pattern"</em>, negative means <em>"bright here? then it's
                probably not me."</em> The <strong>bias</strong> is its threshold: how much persuasion
                it needs before it speaks up at all.</p>
            <p>Drag the sliders. The wires thicken as weights strengthen and flip color when they turn
                negative. The opinion at the end is a single number between 0 and 1.</p>
            <div class="card">
                <div class="neuron-grid">
                    <div>
                        <div class="srow"><label for="nx1">input x₁</label><input type="range" id="nx1" min="0" max="1" step="0.05" value="0.8"><output id="nx1-v">0.80</output></div>
                        <div class="srow"><label for="nw1">weight w₁</label><input type="range" id="nw1" min="-2" max="2" step="0.05" value="1.2"><output id="nw1-v">1.20</output></div>
                        <div class="srow"><label for="nx2">input x₂</label><input type="range" id="nx2" min="0" max="1" step="0.05" value="0.3"><output id="nx2-v">0.30</output></div>
                        <div class="srow"><label for="nw2">weight w₂</label><input type="range" id="nw2" min="-2" max="2" step="0.05" value="-0.6"><output id="nw2-v">-0.60</output></div>
                        <div class="srow"><label for="nb">bias b</label><input type="range" id="nb" min="-2" max="2" step="0.05" value="0.1"><output id="nb-v">0.10</output></div>
                    </div>
                    <div>
                        <svg id="neuron-svg" viewBox="0 0 380 220" role="img" aria-label="diagram of one neuron: two inputs with weighted connections feeding an output">
                            <line id="nl1" x1="58" y1="62" x2="172" y2="110" stroke="#8b7cff" stroke-width="3" stroke-linecap="round"/>
                            <line id="nl2" x1="58" y1="158" x2="172" y2="110" stroke="#ff6f9c" stroke-width="2" stroke-linecap="round"/>
                            <line id="nlo" x1="228" y1="110" x2="318" y2="110" stroke="#ffc466" stroke-width="2" stroke-linecap="round"/>
                            <circle cx="44" cy="62" r="17" fill="#15151f" stroke="rgba(255,255,255,0.25)"/>
                            <circle cx="44" cy="158" r="17" fill="#15151f" stroke="rgba(255,255,255,0.25)"/>
                            <circle id="ncell" cx="200" cy="110" r="29" fill="rgba(139,124,255,0.25)" stroke="#8b7cff" stroke-width="1.5"/>
                            <circle id="nout" cx="334" cy="110" r="15" fill="rgba(255,196,102,0.3)" stroke="#ffc466" stroke-width="1.5"/>
                            <text id="nx1-t" x="44" y="66" text-anchor="middle" font-size="11" fill="#eceae4" font-family="JetBrains Mono, monospace">0.8</text>
                            <text id="nx2-t" x="44" y="162" text-anchor="middle" font-size="11" fill="#eceae4" font-family="JetBrains Mono, monospace">0.3</text>
                            <text id="nz-t" x="200" y="114" text-anchor="middle" font-size="12" fill="#eceae4" font-family="JetBrains Mono, monospace">Σ</text>
                            <text id="na-t" x="334" y="114" text-anchor="middle" font-size="10" fill="#eceae4" font-family="JetBrains Mono, monospace">0.7</text>
                            <text x="44" y="36" text-anchor="middle" font-size="9" fill="#8e8c99" font-family="JetBrains Mono, monospace">inputs</text>
                            <text x="200" y="64" text-anchor="middle" font-size="9" fill="#8e8c99" font-family="JetBrains Mono, monospace">weighted sum + bias</text>
                            <text x="334" y="84" text-anchor="middle" font-size="9" fill="#8e8c99" font-family="JetBrains Mono, monospace">fires</text>
                        </svg>
                        <div id="neuron-formula">…</div>
                    </div>
                </div>
            </div>
            <p style="margin-top:22px">One more move, easy to miss and impossible to skip: the
                <strong>squash</strong> at the end is not linear — it has a bend in it. Without that bend,
                stacking layers would be pointless: a tower of purely linear steps always collapses into
                one linear step, no matter how tall you build it. <em>The bend is where the depth becomes
                real</em> — it lets 64 simple opinions combine into judgements none of them could make alone.</p>
        </section>

        <section class="chapter reveal">
            <span class="chapnum">chapter 03</span>
            <h2>sixty-four pattern hunters</h2>
            <p>The 784 numbers feed <span class="mono">64</span> hidden neurons, which feed
                <span class="mono">10</span> outputs — one per digit, loudest wins. Before you look below,
                make a guess: <em>if you had 64 helpers to recognize digits, what would you tell them to
                hunt for?</em> Loops? Vertical strokes? Corners?</p>
            <p>Nobody told these sixty-four anything. Each tile below is <strong>one hidden neuron's
                actual weights</strong>, lifted from the network running on this page and arranged back
                into a 28×28 image — violet pixels excite it, pink pixels veto it. Purely from being
                graded on its guesses, it invented its own stroke- and loop-detectors. Hover to zoom.</p>
            <div class="card">
                <div class="features-row" id="features"></div>
                <p class="legend">one tile = one hidden neuron's 784 weights
                    <i style="background:#8b7cff"></i>excites <i style="background:#ff6f9c"></i>inhibits</p>
            </div>
        </section>

        <section class="chapter reveal">
            <span class="chapnum">chapter 04</span>
            <h2>learning is a climb down in the dark</h2>
            <p>Imagine standing on a mountainside at midnight with a dying flashlight. You can't see the
                valley — only the tilt of the ground under your feet. So you do the only sensible thing:
                feel the slope, step downhill, repeat. That is <strong>gradient descent</strong>. The
                mountain's altitude is the <strong>loss</strong> — how wrong the network is — and the
                landscape has 50,890 dimensions, one per weight.</p>
            <p>How does each weight know its own slope? After every wrong guess, the error flows
                <em>backwards through the very same wires the signal came forward through</em>. Every weight
                receives blame in proportion to how much it contributed to the mistake — then nudges itself
                the opposite way. That backwards flow of blame is <strong>backpropagation</strong>: the chain
                rule from calculus, wearing work clothes.</p>
            <p>Below, one weight rolls down its own little valley. The <strong>learning rate</strong> is the
                stride length. Push it past <span class="mono">0.9</span> and watch the climber leap clean
                over the valley floor, again and again.</p>
            <div class="card">
                <canvas id="bowl" width="980" height="300"></canvas>
                <div class="bowl-controls">
                    <button id="step-btn">take a step</button>
                    <button id="auto-btn">auto-descend</button>
                    <button id="drop-btn">drop again</button>
                    <label class="mono" for="lr" style="margin-left:8px; color: var(--mut);">lr <span id="lr-v" style="color: var(--ink);">0.20</span></label>
                    <input type="range" id="lr" min="0.05" max="1.05" step="0.05" value="0.2" style="width:130px">
                    <span id="bowl-stats"></span>
                </div>
                <p id="lr-warn"></p>
            </div>
        </section>

        <section class="chapter wide reveal">
            <span class="chapnum">chapter 05 — the lab</span>
            <h2>draw. watch it think. correct it.</h2>
            <p style="max-width:740px">Draw a digit. When you lift your pen, your sketch becomes the 784
                numbers from chapter one, and you'll watch the signal physically flow through the network —
                every glowing wire is a real weighted connection firing right now, in Rust, inside this page.</p>
            <p style="max-width:740px">One honest detail: this network grew up on 60,000 digits written by
                American census workers and high-school students in the 1980s. <em>Your 4 is not their 4.</em>
                When it stumbles on your handwriting, that isn't stupidity — it's a foreign accent.
                <strong>Correct it</strong>, and watch the pink flash of blame run backwards through the wires
                as it adjusts to you. It remembers between visits.</p>
            <div class="card">
                <div class="lab-grid">
                    <div class="pad-side">
                        <canvas id="pad" width="320" height="320" aria-label="drawing pad: draw a digit from 0 to 9"></canvas>
                        <div class="pad-tools">
                            <button id="clear-btn">clear</button>
                            <span id="confidence"></span>
                        </div>
                        <div class="verdict-box">
                            <div id="verdict" class="idle">draw a digit…</div>
                        </div>
                    </div>
                    <div style="min-width:0; width:100%">
                        <p class="viz-label">the forward pass, live</p>
                        <canvas id="netviz"></canvas>
                        <div id="teach-row">
                            <p class="viz-label" style="margin-top:18px">what did you actually write? click to teach it</p>
                            <div class="digit-btns">
                                <button data-digit="0">0</button><button data-digit="1">1</button><button data-digit="2">2</button><button data-digit="3">3</button><button data-digit="4">4</button><button data-digit="5">5</button><button data-digit="6">6</button><button data-digit="7">7</button><button data-digit="8">8</button><button data-digit="9">9</button>
                            </div>
                            <p id="teach-note"></p>
                            <div id="spark-wrap">
                                <p class="viz-label">your lessons — loss per correction (falling = learning)</p>
                                <canvas id="loss-spark" width="480" height="96"></canvas>
                            </div>
                        </div>
                        <div class="teach-meta">
                            <span>taught: <span id="taught-count">0</span></span>
                            <button id="reset-btn">restore factory brain</button>
                            <span id="restored-note"></span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section class="chapter reveal">
            <span class="chapnum">chapter 06</span>
            <h2>the whole secret, in twenty lines</h2>
            <p>Everything above scales. The network in this page was trained with exactly this loop —
                guess, measure the wrongness, let the blame flow backwards, step downhill. Repeated
                840,000 times. The largest models on Earth are this same loop with more zeros attached.</p>
            <div class="card" style="padding: 6px;">
<pre><code><span class="k">import</span> torch, torch.nn <span class="k">as</span> nn
<span class="k">from</span> torchvision <span class="k">import</span> datasets, transforms

train  = datasets.MNIST(<span class="s">'data'</span>, train=<span class="k">True</span>, download=<span class="k">True</span>,
                        transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(train, batch_size=<span class="n">32</span>, shuffle=<span class="k">True</span>)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(<span class="n">28</span> * <span class="n">28</span>, <span class="n">64</span>),   <span class="c"># 784 numbers -> 64 pattern hunters</span>
    nn.ReLU(),                <span class="c"># the bend that makes depth real</span>
    nn.Linear(<span class="n">64</span>, <span class="n">10</span>),        <span class="c"># 64 hunters -> 10 verdicts</span>
)

opt     = torch.optim.SGD(model.parameters(), lr=<span class="n">0.05</span>)
loss_fn = nn.CrossEntropyLoss()

<span class="k">for</span> epoch <span class="k">in</span> range(<span class="n">14</span>):
    <span class="k">for</span> x, y <span class="k">in</span> loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)   <span class="c"># how wrong are we?</span>
        loss.backward()               <span class="c"># blame flows backwards</span>
        opt.step()                    <span class="c"># every weight steps downhill</span></code></pre>
            </div>
        </section>

        <section class="epilogue reveal">
            <p class="big">A pile of weighted opinions, a bend that makes depth real,
                and blame that flows backwards until the guesses stop being wrong.<br>
                You just taught one by hand.</p>
        </section>

        <footer>
            <a href="../index.html">&larr; back home</a> &nbsp;·&nbsp; © Journey to the stars🚀🌟
        </footer>
    </main>"##;

    head + chapters
}
