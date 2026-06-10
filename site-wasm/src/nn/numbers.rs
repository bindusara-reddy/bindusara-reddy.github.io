//! chapter 01: draw a "7" through the real preprocessing pipeline and show
//! it the way the network receives it — as a grid of numbers.

use crate::util::*;
use wasm_bindgen::JsCast;

pub fn init() {
    // draw a 7 on an offscreen pad, exactly like a visitor would
    let doc = document();
    let pad: web_sys::HtmlCanvasElement =
        doc.create_element("canvas").unwrap().dyn_into().unwrap();
    pad.set_width(280);
    pad.set_height(280);
    let ctx = ctx2d(&pad);
    fill_style(&ctx, "#000");
    ctx.fill_rect(0.0, 0.0, 280.0, 280.0);
    stroke_style(&ctx, "#fff");
    ctx.set_line_width(20.0);
    ctx.set_line_cap("round");
    ctx.set_line_join("round");
    ctx.begin_path();
    ctx.move_to(70.0, 60.0);
    ctx.line_to(205.0, 58.0);
    ctx.line_to(120.0, 235.0);
    ctx.stroke();

    let x = crate::nn::lab::canvas_to_784(&pad, 0).unwrap_or_else(|| vec![0.0; 784]);

    // render the 28x28 grid of values
    let cv = get_canvas("numbers");
    let c = ctx2d(&cv);
    let cell = cv.width() as f64 / 28.0; // square canvas
    c.set_text_align("center");
    for row in 0..28 {
        for col in 0..28 {
            let v = x[row * 28 + col] as f64;
            let px = col as f64 * cell;
            let py = row as f64 * cell;
            fill_style(&c, &format!("rgba(139,124,255,{:.3})", v * 0.32));
            c.fill_rect(px, py, cell, cell);
            let d = (v * 9.0).round() as i32;
            if d > 0 {
                fill_style(&c, &format!("rgba(236,234,228,{:.2})", 0.25 + 0.75 * v));
                c.set_font(&format!("{}px JetBrains Mono, monospace", (cell * 0.5) as i32));
                let _ = c.fill_text(&d.to_string(), px + cell / 2.0, py + cell * 0.68);
            } else {
                fill_style(&c, "rgba(142,140,153,0.16)");
                c.set_font(&format!("{}px JetBrains Mono, monospace", (cell * 0.4) as i32));
                let _ = c.fill_text("0", px + cell / 2.0, py + cell * 0.66);
            }
        }
    }
}
