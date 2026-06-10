//! chapter 05: the lab. draw -> preprocess -> animated forward pass ->
//! teach (backprop) with a backwards blame flash -> loss sparkline.

use crate::engine::{bump_taught, taught_count, Net, OUT};
use crate::util::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

const VW: f64 = 660.0;
const VH: f64 = 420.0;

// layout
const THUMB: (f64, f64, f64) = (16.0, 162.0, 96.0); // x, y, size
const GRID: (f64, f64, f64, f64) = (178.0, 152.0, 16.5, 4.5); // x, y, gap, r
const OUT_X: f64 = 380.0;
const OUT_Y0: f64 = 30.0;
const OUT_DY: f64 = 38.5;
const BAR0: f64 = 424.0;
const BAR1: f64 = 588.0;

pub struct Lab {
    net: Net,
    x: Option<Vec<f32>>,
    a1: Vec<f32>,
    probs: [f32; OUT],
    winner: usize,
    drawing: bool,
    last: (f64, f64),
    has_ink: bool,
    losses: Vec<f32>,
    token: u32,
    blame_from: Option<usize>,
}

type L = Rc<RefCell<Lab>>;

/// shared with chapter 01: shrink any canvas of white-on-dark ink to the
/// MNIST 28x28 retina (crop, scale to 20, recentre by centre of mass)
pub fn canvas_to_784(src: &HtmlCanvasElement, threshold: u8) -> Option<Vec<f32>> {
    let w = src.width() as usize;
    let h = src.height() as usize;
    let ctx = ctx2d(src);
    let data = ctx
        .get_image_data(0.0, 0.0, w as f64, h as f64)
        .ok()?
        .data();
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (w as i32, h as i32, -1i32, -1i32);
    for y in 0..h {
        for x in 0..w {
            if data[(y * w + x) * 4] > threshold {
                min_x = min_x.min(x as i32);
                max_x = max_x.max(x as i32);
                min_y = min_y.min(y as i32);
                max_y = max_y.max(y as i32);
            }
        }
    }
    if max_x < 0 {
        return None;
    }
    let (bw, bh) = ((max_x - min_x + 1) as f64, (max_y - min_y + 1) as f64);
    let scale = 20.0 / bw.max(bh);
    let sw = (bw * scale).round().max(1.0);
    let sh = (bh * scale).round().max(1.0);

    let doc = document();
    let tmp: HtmlCanvasElement = doc.create_element("canvas").ok()?.dyn_into().ok()?;
    tmp.set_width(28);
    tmp.set_height(28);
    let tctx = ctx2d(&tmp);
    fill_style(&tctx, "#000");
    tctx.fill_rect(0.0, 0.0, 28.0, 28.0);
    tctx.draw_image_with_html_canvas_element_and_sw_and_sh_and_dx_and_dy_and_dw_and_dh(
        src,
        min_x as f64,
        min_y as f64,
        bw,
        bh,
        ((28.0 - sw) / 2.0).floor(),
        ((28.0 - sh) / 2.0).floor(),
        sw,
        sh,
    )
    .ok()?;

    let td = tctx.get_image_data(0.0, 0.0, 28.0, 28.0).ok()?.data();
    let (mut m, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..28 {
        for x in 0..28 {
            let v = td[(y * 28 + x) * 4] as f64;
            m += v;
            mx += x as f64 * v;
            my += y as f64 * v;
        }
    }
    if m <= 0.0 {
        return None;
    }
    let dx = (13.5 - mx / m).round();
    let dy = (13.5 - my / m).round();
    let out: HtmlCanvasElement = doc.create_element("canvas").ok()?.dyn_into().ok()?;
    out.set_width(28);
    out.set_height(28);
    let octx = ctx2d(&out);
    fill_style(&octx, "#000");
    octx.fill_rect(0.0, 0.0, 28.0, 28.0);
    octx.draw_image_with_html_canvas_element(&tmp, dx, dy).ok()?;
    let od = octx.get_image_data(0.0, 0.0, 28.0, 28.0).ok()?.data();
    Some((0..784).map(|i| od[i * 4] as f32 / 255.0).collect())
}

fn dot_pos(h: usize) -> (f64, f64) {
    (GRID.0 + (h % 8) as f64 * GRID.2, GRID.1 + (h / 8) as f64 * GRID.2)
}

fn round_bar(ctx: &web_sys::CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64) {
    let r = h / 2.0;
    ctx.begin_path();
    ctx.move_to(x + r, y);
    let _ = ctx.arc_to(x + w, y, x + w, y + h, r);
    let _ = ctx.arc_to(x + w, y + h, x, y + h, r);
    let _ = ctx.arc_to(x, y + h, x, y, r);
    let _ = ctx.arc_to(x, y, x + w, y, r);
    ctx.fill();
}

fn render_viz(lab: &Lab, t: f64, blame_t: f64) {
    let viz = get_canvas("netviz");
    let ctx = ctx2d(&viz);
    ctx.save();
    let _ = ctx.scale(2.0, 2.0);
    ctx.clear_rect(0.0, 0.0, VW, VH);

    let t1 = ease_out(clamp01(t / 0.4));
    let t2 = ease_out(clamp01((t - 0.28) / 0.4));
    let t3 = ease_out(clamp01((t - 0.5) / 0.5));
    let has = lab.x.is_some();

    // input thumbnail
    stroke_style(&ctx, "rgba(139,124,255,0.4)");
    ctx.set_line_width(1.0);
    ctx.stroke_rect(THUMB.0, THUMB.1, THUMB.2, THUMB.2);
    if let Some(x) = &lab.x {
        let cell = THUMB.2 / 28.0;
        for i in 0..784 {
            let v = x[i] as f64;
            if v < 0.04 {
                continue;
            }
            fill_style(&ctx, &format!("rgba(185,173,255,{v:.2})"));
            ctx.fill_rect(
                THUMB.0 + (i % 28) as f64 * cell,
                THUMB.1 + (i / 28) as f64 * cell,
                cell + 0.5,
                cell + 0.5,
            );
        }
    } else {
        fill_style(&ctx, "rgba(142,140,153,0.5)");
        ctx.set_font("10px JetBrains Mono, monospace");
        let _ = ctx.fill_text("28×28", THUMB.0 + 30.0, THUMB.1 + THUMB.2 / 2.0 + 3.0);
    }
    fill_style(&ctx, "rgba(142,140,153,0.75)");
    ctx.set_font("10px JetBrains Mono, monospace");
    let _ = ctx.fill_text("your digit", THUMB.0 + 14.0, THUMB.1 + THUMB.2 + 18.0);
    let _ = ctx.fill_text("784 inputs", THUMB.0 + 14.0, THUMB.1 + THUMB.2 + 32.0);

    let max_a = lab.a1.iter().cloned().fold(0.0f32, f32::max);

    // input -> hidden wires
    if has && max_a > 0.0 {
        let (sx, sy) = (THUMB.0 + THUMB.2, THUMB.1 + THUMB.2 / 2.0);
        for h in 0..lab.a1.len() {
            let act = (lab.a1[h] / max_a) as f64;
            if act < 0.05 {
                continue;
            }
            let (px, py) = dot_pos(h);
            stroke_style(&ctx, &format!("rgba(139,124,255,{:.3})", act * 0.34));
            ctx.set_line_width(0.8);
            ctx.begin_path();
            ctx.move_to(sx, sy);
            ctx.line_to(sx + (px - sx) * t1, sy + (py - sy) * t1);
            ctx.stroke();
        }
    }

    // hidden dots
    for h in 0..64 {
        let (px, py) = dot_pos(h);
        let act = if has && max_a > 0.0 { (lab.a1[h] / max_a) as f64 } else { 0.0 };
        let pop = if has { t1 } else { 1.0 };
        ctx.begin_path();
        let _ = ctx.arc(px, py, GRID.3 * (0.7 + 0.5 * act * pop), 0.0, std::f64::consts::TAU);
        fill_style(&ctx, &format!("rgba(185,173,255,{:.3})", 0.12 + 0.88 * act * pop));
        ctx.fill();
    }
    fill_style(&ctx, "rgba(142,140,153,0.75)");
    ctx.set_font("10px JetBrains Mono, monospace");
    let _ = ctx.fill_text("64 hidden neurons", GRID.0 - 4.0, GRID.1 + 7.0 * GRID.2 + 26.0);

    // hidden -> output wires
    if has && t2 > 0.0 {
        let w2 = lab.net.w2();
        let hid = lab.a1.len();
        for o in 0..OUT {
            let mut contribs: Vec<(f32, f32, usize)> = (0..hid)
                .filter_map(|h| {
                    let c = lab.a1[h] * w2[o * hid + h];
                    if c != 0.0 { Some((c.abs(), c, h)) } else { None }
                })
                .collect();
            contribs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            contribs.truncate(6);
            let max_c = contribs.first().map(|c| c.0).unwrap_or(1.0);
            let oy = OUT_Y0 + o as f64 * OUT_DY;
            for (ac, c, h) in &contribs {
                let (px, py) = dot_pos(*h);
                let k = (ac / max_c) as f64;
                let base = if o == lab.winner { 0.55 } else { 0.10 };
                let style = if *c >= 0.0 {
                    if o == lab.winner {
                        format!("rgba(255,196,102,{:.3})", base * k)
                    } else {
                        format!("rgba(139,124,255,{:.3})", base * k)
                    }
                } else {
                    format!("rgba(255,111,156,{:.3})", base * k * 0.9)
                };
                stroke_style(&ctx, &style);
                ctx.set_line_width(if o == lab.winner { 1.3 } else { 0.8 });
                ctx.begin_path();
                ctx.move_to(px, py);
                ctx.line_to(px + (OUT_X - px) * t2, py + (oy - py) * t2);
                ctx.stroke();
            }
        }
    }

    // blame flash: error flowing backwards through the wires, in pink
    if blame_t > 0.0 {
        if let Some(label) = lab.blame_from {
            let a = (1.0 - blame_t) * 0.9;
            let oy = OUT_Y0 + label as f64 * OUT_DY;
            let hid = lab.a1.len();
            for h in 0..hid {
                if max_a > 0.0 && (lab.a1[h] / max_a) as f64 > 0.12 {
                    let (px, py) = dot_pos(h);
                    // sweep from output back to hidden
                    let bx = OUT_X + (px - OUT_X) * blame_t;
                    let by = oy + (py - oy) * blame_t;
                    stroke_style(&ctx, &format!("rgba(255,111,156,{:.3})", a * 0.6));
                    ctx.set_line_width(1.4);
                    ctx.begin_path();
                    ctx.move_to(OUT_X, oy);
                    ctx.line_to(bx, by);
                    ctx.stroke();
                }
            }
            let (sx, sy) = (THUMB.0 + THUMB.2, THUMB.1 + THUMB.2 / 2.0);
            let gt = clamp01(blame_t * 1.4 - 0.3);
            if gt > 0.0 {
                for h in 0..hid {
                    if max_a > 0.0 && (lab.a1[h] / max_a) as f64 > 0.25 {
                        let (px, py) = dot_pos(h);
                        stroke_style(&ctx, &format!("rgba(255,111,156,{:.3})", a * 0.35));
                        ctx.set_line_width(1.0);
                        ctx.begin_path();
                        ctx.move_to(px, py);
                        ctx.line_to(px + (sx - px) * gt, py + (sy - py) * gt);
                        ctx.stroke();
                    }
                }
            }
        }
    }

    // output nodes, labels, bars
    for o in 0..OUT {
        let oy = OUT_Y0 + o as f64 * OUT_DY;
        let prob = if has { lab.probs[o] as f64 } else { 0.0 };
        let win = has && o == lab.winner;
        let lit = prob * t3;

        ctx.begin_path();
        let _ = ctx.arc(OUT_X, oy, 7.0, 0.0, std::f64::consts::TAU);
        fill_style(
            &ctx,
            &if win {
                format!("rgba(255,196,102,{:.2})", 0.2 + 0.8 * t3)
            } else {
                format!("rgba(185,173,255,{:.2})", 0.1 + 0.55 * lit)
            },
        );
        ctx.fill();

        ctx.set_font(if win { "700 13px JetBrains Mono, monospace" } else { "13px JetBrains Mono, monospace" });
        fill_style(&ctx, if win { "#ffc466" } else { "rgba(201,199,210,0.8)" });
        let _ = ctx.fill_text(&o.to_string(), 402.0, oy + 4.5);

        fill_style(&ctx, "rgba(255,255,255,0.06)");
        round_bar(&ctx, BAR0, oy - 5.5, BAR1 - BAR0, 11.0);
        if lit > 0.002 {
            fill_style(&ctx, if win { "#ffc466" } else { "rgba(139,124,255,0.75)" });
            round_bar(&ctx, BAR0, oy - 5.5, ((BAR1 - BAR0) * lit).max(3.0), 11.0);
        }
        ctx.set_font("11px JetBrains Mono, monospace");
        fill_style(&ctx, if win { "rgba(255,196,102,0.95)" } else { "rgba(142,140,153,0.85)" });
        let _ = ctx.fill_text(&format!("{}%", (prob * t3 * 100.0).round()), 596.0, oy + 4.0);
    }

    ctx.restore();
}

fn animate(lab_rc: &L, with_blame: bool) {
    let tok = {
        let mut l = lab_rc.borrow_mut();
        l.token = l.token.wrapping_add(1);
        l.token
    };
    if reduced_motion() {
        render_viz(&lab_rc.borrow(), 1.0, 0.0);
        return;
    }
    let lab_rc = lab_rc.clone();
    let t0 = now();
    let blame_dur = if with_blame { 420.0 } else { 0.0 };
    let fwd_dur = 850.0;
    raf_loop(move |_| {
        let l = lab_rc.borrow();
        if l.token != tok {
            return false;
        }
        let el = now() - t0;
        if el < blame_dur {
            render_viz(&l, 1.0, clamp01(el / blame_dur));
            true
        } else {
            let t = clamp01((el - blame_dur) / fwd_dur);
            render_viz(&l, t, 0.0);
            t < 1.0
        }
    });
}

fn apply_prediction(lab_rc: &L) {
    let (x, has) = {
        let l = lab_rc.borrow();
        (l.x.clone(), l.x.is_some())
    };
    if !has {
        return;
    }
    let x = x.unwrap();
    {
        let mut l = lab_rc.borrow_mut();
        let (a1, probs) = l.net.forward(&x);
        let mut winner = 0;
        for d in 1..OUT {
            if probs[d] > probs[winner] {
                winner = d;
            }
        }
        l.a1 = a1;
        l.probs = probs;
        l.winner = winner;
    }
    let l = lab_rc.borrow();
    let v = get("verdict");
    v.set_text_content(Some(&l.winner.to_string()));
    let _ = v.class_list().remove_1("idle");
    let conf = l.probs[l.winner];
    set_text(
        "confidence",
        &if conf > 0.85 {
            format!("{}% sure", (conf * 100.0).round())
        } else if conf > 0.5 {
            format!("{}% sure… probably", (conf * 100.0).round())
        } else {
            format!("{}% sure — teach me?", (conf * 100.0).round())
        },
    );
    let _ = get("teach-row").class_list().add_1("armed");
}

fn predict(lab_rc: &L) {
    let pad = get_canvas("pad");
    let Some(x) = canvas_to_784(&pad, 40) else { return };
    lab_rc.borrow_mut().x = Some(x);
    lab_rc.borrow_mut().blame_from = None;
    apply_prediction(lab_rc);
    animate(lab_rc, false);
}

fn draw_spark(losses: &[f32]) {
    let cv = get_canvas("loss-spark");
    let c = ctx2d(&cv);
    let (w, h) = (cv.width() as f64, cv.height() as f64);
    c.clear_rect(0.0, 0.0, w, h);
    if losses.is_empty() {
        return;
    }
    let data: Vec<f32> = losses.iter().rev().take(30).rev().cloned().collect();
    let mx = data.iter().cloned().fold(0.5f32, f32::max) as f64;
    stroke_style(&c, "rgba(255,255,255,0.12)");
    c.begin_path();
    c.move_to(0.0, h - 8.0);
    c.line_to(w, h - 8.0);
    c.stroke();
    let pt = |i: usize, l: f32| -> (f64, f64) {
        let x = if data.len() == 1 {
            w / 2.0
        } else {
            8.0 + i as f64 / (data.len() - 1) as f64 * (w - 16.0)
        };
        (x, h - 8.0 - (l as f64 / mx) * (h - 20.0))
    };
    stroke_style(&c, "#ffc466");
    c.set_line_width(2.5);
    c.begin_path();
    for (i, l) in data.iter().enumerate() {
        let (x, y) = pt(i, *l);
        if i == 0 {
            c.move_to(x, y);
        } else {
            c.line_to(x, y);
        }
    }
    c.stroke();
    fill_style(&c, "#ffc466");
    for (i, l) in data.iter().enumerate() {
        let (x, y) = pt(i, *l);
        c.begin_path();
        let _ = c.arc(x, y, 3.5, 0.0, std::f64::consts::TAU);
        c.fill();
    }
}

fn teach(lab_rc: &L, label: usize) {
    let has = lab_rc.borrow().x.is_some();
    if !has {
        return;
    }
    let loss = {
        let mut l = lab_rc.borrow_mut();
        let x = l.x.clone().unwrap();
        let loss = l.net.train(&x, label, 0.012, 4);
        l.losses.push(loss);
        l.blame_from = Some(label);
        loss
    };
    let n = bump_taught();
    set_text("taught-count", &n.to_string());
    let _ = get("spark-wrap").class_list().add_1("show");
    draw_spark(&lab_rc.borrow().losses);
    set_text(
        "teach-note",
        &format!("learned. loss {loss:.3} — pink = blame flowing backwards"),
    );
    apply_prediction(lab_rc);
    animate(lab_rc, true);
}

fn clear_pad(lab_rc: &L) {
    let pad = get_canvas("pad");
    let ctx = ctx2d(&pad);
    fill_style(&ctx, "#0d0d14");
    ctx.fill_rect(0.0, 0.0, 320.0, 320.0);
    {
        let mut l = lab_rc.borrow_mut();
        l.x = None;
        l.has_ink = false;
        l.blame_from = None;
        l.a1 = vec![0.0; l.a1.len()];
    }
    let v = get("verdict");
    v.set_text_content(Some("draw a digit…"));
    let _ = v.class_list().add_1("idle");
    set_text("confidence", "");
    let _ = get("teach-row").class_list().remove_1("armed");
    render_viz(&lab_rc.borrow(), 1.0, 0.0);
}

pub fn init() {
    let (net, restored) = Net::load();
    let hid = net.hid;
    let lab: L = Rc::new(RefCell::new(Lab {
        net,
        x: None,
        a1: vec![0.0; hid],
        probs: [0.0; OUT],
        winner: 0,
        drawing: false,
        last: (0.0, 0.0),
        has_ink: false,
        losses: Vec::new(),
        token: 0,
        blame_from: None,
    }));

    let viz = get_canvas("netviz");
    viz.set_width((VW * 2.0) as u32);
    viz.set_height((VH * 2.0) as u32);

    let pad = get_canvas("pad");

    let pos = {
        let pad = pad.clone();
        move |e: &web_sys::PointerEvent| -> (f64, f64) {
            let r = pad.get_bounding_client_rect();
            (
                (e.client_x() as f64 - r.left()) * pad.width() as f64 / r.width(),
                (e.client_y() as f64 - r.top()) * pad.height() as f64 / r.height(),
            )
        }
    };

    {
        let lab = lab.clone();
        let pad2 = pad.clone();
        let pos = pos.clone();
        on(pad.as_ref(), "pointerdown", move |e| {
            let pe: web_sys::PointerEvent = e.dyn_into().unwrap();
            pe.prevent_default();
            let _ = pad2.set_pointer_capture(pe.pointer_id());
            let mut l = lab.borrow_mut();
            l.drawing = true;
            l.last = pos(&pe);
        });
    }
    {
        let lab = lab.clone();
        let pad2 = pad.clone();
        let pos = pos.clone();
        on(pad.as_ref(), "pointermove", move |e| {
            let pe: web_sys::PointerEvent = e.dyn_into().unwrap();
            let mut l = lab.borrow_mut();
            if !l.drawing {
                return;
            }
            let p = pos(&pe);
            let ctx = ctx2d(&pad2);
            stroke_style(&ctx, "#fff");
            ctx.set_line_width(22.0);
            ctx.set_line_cap("round");
            ctx.set_line_join("round");
            ctx.begin_path();
            ctx.move_to(l.last.0, l.last.1);
            ctx.line_to(p.0, p.1);
            ctx.stroke();
            l.last = p;
            l.has_ink = true;
        });
    }
    for ev in ["pointerup", "pointercancel"] {
        let lab = lab.clone();
        on(pad.as_ref(), ev, move |_| {
            let was = {
                let mut l = lab.borrow_mut();
                let was = l.drawing && l.has_ink;
                l.drawing = false;
                was
            };
            if was {
                predict(&lab);
            }
        });
    }

    {
        let lab = lab.clone();
        on_click("clear-btn", move || clear_pad(&lab));
    }
    for d in 0..10usize {
        let lab = lab.clone();
        let btns = document().query_selector_all("#teach-row button[data-digit]").unwrap();
        let el: web_sys::Element = btns.item(d as u32).unwrap().dyn_into().unwrap();
        on(el.as_ref(), "click", move |_| teach(&lab, d));
    }
    {
        let lab = lab.clone();
        on_click("reset-btn", move || {
            if window()
                .confirm_with_message("Forget everything you taught it and restore the factory brain?")
                .unwrap_or(false)
            {
                {
                    let mut l = lab.borrow_mut();
                    l.net.reset();
                    l.losses.clear();
                }
                set_text("taught-count", "0");
                set_text("teach-note", "");
                let _ = get("spark-wrap").class_list().remove_1("show");
                if lab.borrow().x.is_some() {
                    apply_prediction(&lab);
                    animate(&lab, false);
                }
            }
        });
    }

    set_text("taught-count", &taught_count().to_string());
    if restored {
        set_text("restored-note", "this browser remembers your lessons");
    }
    clear_pad(&lab);
}
