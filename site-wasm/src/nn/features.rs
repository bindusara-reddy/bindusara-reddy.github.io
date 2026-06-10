//! chapter 03: render what the hidden neurons actually learned

use crate::engine::Net;
use crate::util::*;
use wasm_bindgen::JsCast;

pub fn init() {
    let net = Net::factory();
    let w1 = net.w1();
    let hid = net.hid;

    let mut norms: Vec<(f32, usize)> = (0..hid)
        .map(|h| {
            let s: f32 = w1[h * 784..(h + 1) * 784].iter().map(|v| v * v).sum();
            (s, h)
        })
        .collect();
    norms.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let host = get("features");
    for k in 0..8 {
        let h = norms[k].1;
        let row = &w1[h * 784..(h + 1) * 784];
        let mx = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let cv: web_sys::HtmlCanvasElement = document()
            .create_element("canvas")
            .unwrap()
            .dyn_into()
            .unwrap();
        cv.set_width(28);
        cv.set_height(28);
        cv.set_class_name("feature");
        let _ = cv.set_attribute("title", &format!("hidden neuron #{h}"));
        let c = ctx2d(&cv);
        for i in 0..784 {
            let v = row[i] / mx;
            let color = if v >= 0.0 {
                format!(
                    "rgb({},{},{})",
                    (139.0 * v + 7.0) as u8,
                    (124.0 * v + 7.0) as u8,
                    (244.0 * v + 11.0) as u8
                )
            } else {
                format!(
                    "rgb({},{},{})",
                    (248.0 * -v + 7.0) as u8,
                    (111.0 * -v + 7.0) as u8,
                    (145.0 * -v + 11.0) as u8
                )
            };
            fill_style(&c, &color);
            c.fill_rect((i % 28) as f64, (i / 28) as f64, 1.0, 1.0);
        }
        host.append_child(&cv).unwrap();
    }
}
