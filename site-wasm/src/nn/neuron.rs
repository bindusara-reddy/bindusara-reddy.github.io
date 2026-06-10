//! chapter 02: the pokeable neuron

use crate::util::*;

const IDS: [&str; 5] = ["nx1", "nw1", "nx2", "nw2", "nb"];

fn val(id: &str) -> f64 {
    get_input(id).value().parse().unwrap_or(0.0)
}

fn update() {
    let (x1, w1, x2, w2, b) = (val("nx1"), val("nw1"), val("nx2"), val("nw2"), val("nb"));
    let z = x1 * w1 + x2 * w2 + b;
    let a = 1.0 / (1.0 + (-z).exp());

    for id in IDS {
        set_text(&format!("{id}-v"), &format!("{:.2}", val(id)));
    }

    let set = |id: &str, attr: &str, v: String| {
        let _ = get(id).set_attribute(attr, &v);
    };
    set("nl1", "stroke-width", format!("{:.2}", 0.6 + w1.abs() * 3.2));
    set("nl2", "stroke-width", format!("{:.2}", 0.6 + w2.abs() * 3.2));
    set("nl1", "stroke", if w1 >= 0.0 { "#8b7cff" } else { "#ff6f9c" }.into());
    set("nl2", "stroke", if w2 >= 0.0 { "#8b7cff" } else { "#ff6f9c" }.into());
    set("nl1", "opacity", format!("{:.2}", 0.35 + (w1.abs() / 2.0).min(1.0) * 0.65));
    set("nl2", "opacity", format!("{:.2}", 0.35 + (w2.abs() / 2.0).min(1.0) * 0.65));
    set("nlo", "stroke-width", format!("{:.2}", 0.5 + a * 5.0));
    set("nlo", "opacity", format!("{:.2}", 0.25 + a * 0.75));
    set("ncell", "fill", format!("rgba(139,124,255,{:.2})", 0.1 + 0.45 * a));
    set("nout", "r", format!("{:.1}", 10.0 + a * 8.0));
    set("nout", "fill", format!("rgba(255,196,102,{:.2})", 0.12 + 0.7 * a));

    set_text("nx1-t", &format!("{x1:.1}"));
    set_text("nx2-t", &format!("{x2:.1}"));
    set_text("nz-t", &format!("{z:.2}"));
    set_text("na-t", &format!("{a:.2}"));
    set_text(
        "neuron-formula",
        &format!("{x1:.2}×{w1:.2} + {x2:.2}×{w2:.2} + {b:.2} = {z:.2}  →  squashed to {a:.2}"),
    );
}

pub fn init() {
    for id in IDS {
        on(get(id).as_ref(), "input", move |_| update());
    }
    update();
}
