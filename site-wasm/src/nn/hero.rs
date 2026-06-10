//! the hero constellation: drifting nodes, faint wires, amber pulses

use crate::util::*;
use std::cell::RefCell;
use std::rc::Rc;

struct P {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

struct Pulse {
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    t: f64,
}

pub fn init() {
    let cv = get_canvas("hero-net");
    let ctx = ctx2d(&cv);

    let state: Rc<RefCell<(Vec<P>, Vec<Pulse>, f64, f64)>> =
        Rc::new(RefCell::new((Vec::new(), Vec::new(), 0.0, 0.0)));

    let seed = {
        let cv = cv.clone();
        let state = state.clone();
        move || {
            let w = cv.client_width() as f64;
            let h = cv.client_height() as f64;
            cv.set_width(w as u32);
            cv.set_height(h as u32);
            let n = (w * h / 16000.0).min(70.0) as usize;
            let nodes = (0..n)
                .map(|_| P {
                    x: random() * w,
                    y: random() * h,
                    vx: (random() - 0.5) * 0.25,
                    vy: (random() - 0.5) * 0.25,
                })
                .collect();
            *state.borrow_mut() = (nodes, Vec::new(), w, h);
        }
    };
    seed();

    {
        let seed = seed.clone();
        on(window().as_ref(), "resize", move |_| seed());
    }

    let draw = {
        let state = state.clone();
        move || {
            let st = state.borrow();
            let (nodes, pulses, w, h) = (&st.0, &st.1, st.2, st.3);
            ctx.clear_rect(0.0, 0.0, w, h);
            let r = 130.0;
            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    let dx = nodes[i].x - nodes[j].x;
                    let dy = nodes[i].y - nodes[j].y;
                    let d2 = dx * dx + dy * dy;
                    if d2 < r * r {
                        let t = 1.0 - d2.sqrt() / r;
                        stroke_style(&ctx, &format!("rgba(139,124,255,{:.3})", t * 0.16));
                        ctx.set_line_width(1.0);
                        ctx.begin_path();
                        ctx.move_to(nodes[i].x, nodes[i].y);
                        ctx.line_to(nodes[j].x, nodes[j].y);
                        ctx.stroke();
                    }
                }
            }
            for n in nodes {
                fill_style(&ctx, "rgba(185,173,255,0.5)");
                ctx.begin_path();
                let _ = ctx.arc(n.x, n.y, 1.6, 0.0, std::f64::consts::TAU);
                ctx.fill();
            }
            for p in pulses {
                let x = p.ax + (p.bx - p.ax) * p.t;
                let y = p.ay + (p.by - p.ay) * p.t;
                fill_style(&ctx, &format!("rgba(255,196,102,{:.3})", (1.0 - p.t) * 0.9));
                ctx.begin_path();
                let _ = ctx.arc(x, y, 2.6, 0.0, std::f64::consts::TAU);
                ctx.fill();
            }
        }
    };

    if reduced_motion() {
        draw();
        return;
    }

    let frame = Rc::new(RefCell::new(0u64));
    raf_loop(move |_| {
        *frame.borrow_mut() += 1;
        if *frame.borrow() % 2 == 0 {
            {
                let mut st = state.borrow_mut();
                let (w, h) = (st.2, st.3);
                for n in st.0.iter_mut() {
                    n.x += n.vx;
                    n.y += n.vy;
                    if n.x < 0.0 || n.x > w {
                        n.vx = -n.vx;
                    }
                    if n.y < 0.0 || n.y > h {
                        n.vy = -n.vy;
                    }
                }
                for p in st.1.iter_mut() {
                    p.t += 0.03;
                }
                st.1.retain(|p| p.t < 1.0);
                if random() < 0.05 && st.1.len() < 6 && st.0.len() > 1 {
                    let i = (random() * st.0.len() as f64) as usize;
                    let (ax, ay) = (st.0[i].x, st.0[i].y);
                    let mut best: Option<(f64, f64)> = None;
                    let mut bd = f64::MAX;
                    for (j, b) in st.0.iter().enumerate() {
                        if j == i {
                            continue;
                        }
                        let d = (ax - b.x).powi(2) + (ay - b.y).powi(2);
                        if d < bd && d < 130.0 * 130.0 {
                            bd = d;
                            best = Some((b.x, b.y));
                        }
                    }
                    if let Some((bx, by)) = best {
                        st.1.push(Pulse { ax, ay, bx, by, t: 0.0 });
                    }
                }
            }
            draw();
        }
        true
    });
}
