//! chapter 04: gradient descent rolling down a loss bowl, with a trail

use crate::util::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

const TARGET: f64 = 1.1;
const W: f64 = 980.0;
const H: f64 = 300.0;

fn loss(v: f64) -> f64 {
    (v - TARGET) * (v - TARGET)
}

fn x_of(v: f64) -> f64 {
    (v + 3.6) / 7.2 * W
}

fn y_of(v: f64) -> f64 {
    H - 36.0 - loss(v) / loss(-3.6) * (H - 80.0)
}

struct Bowl {
    w: f64,
    trail: Vec<f64>,
    timer: Option<i32>,
}

pub fn init() {
    let cv = get_canvas("bowl");
    let ctx = ctx2d(&cv);
    let st = Rc::new(RefCell::new(Bowl { w: -2.7, trail: Vec::new(), timer: None }));

    let draw = {
        let st = st.clone();
        move || {
            let s = st.borrow();
            ctx.clear_rect(0.0, 0.0, W, H);
            stroke_style(&ctx, "rgba(139,124,255,0.9)");
            ctx.set_line_width(2.5);
            ctx.begin_path();
            let mut px = 0.0;
            while px <= W {
                let v = px / W * 7.2 - 3.6;
                if px == 0.0 {
                    ctx.move_to(px, y_of(v));
                } else {
                    ctx.line_to(px, y_of(v));
                }
                px += 6.0;
            }
            ctx.stroke();
            fill_style(&ctx, "rgba(142,140,153,0.8)");
            ctx.set_font("15px JetBrains Mono, monospace");
            let _ = ctx.fill_text("loss", 16.0, 28.0);
            let _ = ctx.fill_text("weight →", W - 110.0, H - 12.0);
            let n = s.trail.len();
            for (i, v) in s.trail.iter().enumerate() {
                let t = (i + 1) as f64 / n as f64;
                fill_style(&ctx, &format!("rgba(255,196,102,{:.2})", t * 0.35));
                ctx.begin_path();
                let _ = ctx.arc(x_of(*v), y_of(*v), 5.0, 0.0, std::f64::consts::TAU);
                ctx.fill();
            }
            ctx.begin_path();
            let _ = ctx.arc(x_of(s.w), y_of(s.w), 11.0, 0.0, std::f64::consts::TAU);
            fill_style(&ctx, "#ffc466");
            ctx.set_shadow_color("rgba(255,196,102,0.9)");
            ctx.set_shadow_blur(26.0);
            ctx.fill();
            ctx.set_shadow_blur(0.0);
            set_text("bowl-stats", &format!("w = {:.2}  loss = {:.3}", s.w, loss(s.w)));
        }
    };

    let step = {
        let st = st.clone();
        let draw = draw.clone();
        move || {
            let lr: f64 = get_input("lr").value().parse().unwrap_or(0.2);
            {
                let mut s = st.borrow_mut();
                let w = s.w;
                s.trail.push(w);
                if s.trail.len() > 26 {
                    s.trail.remove(0);
                }
                s.w -= lr * 2.0 * (s.w - TARGET);
                s.w = s.w.clamp(-3.5, 3.5);
            }
            set_text(
                "lr-warn",
                if lr > 0.9 { "⚠ stride too long — it keeps leaping over the valley" } else { "" },
            );
            draw();
        }
    };

    {
        let step = step.clone();
        on_click("step-btn", move || step());
    }
    {
        let st = st.clone();
        let step = step.clone();
        on_click("auto-btn", move || {
            let btn = get("auto-btn");
            let mut s = st.borrow_mut();
            if let Some(id) = s.timer.take() {
                window().clear_interval_with_handle(id);
                btn.set_text_content(Some("auto-descend"));
            } else {
                let step = step.clone();
                let cb = Closure::wrap(Box::new(move || step()) as Box<dyn FnMut()>);
                let id = window()
                    .set_interval_with_callback_and_timeout_and_arguments_0(
                        cb.as_ref().unchecked_ref(),
                        260,
                    )
                    .unwrap();
                cb.forget();
                s.timer = Some(id);
                btn.set_text_content(Some("stop"));
            }
        });
    }
    {
        let st = st.clone();
        let draw = draw.clone();
        on_click("drop-btn", move || {
            {
                let mut s = st.borrow_mut();
                s.w = -2.7 + random() * 0.9;
                s.trail.clear();
            }
            draw();
        });
    }
    on(get("lr").as_ref(), "input", |_| {
        let v: f64 = get_input("lr").value().parse().unwrap_or(0.2);
        set_text("lr-v", &format!("{v:.2}"));
    });

    draw();
}
