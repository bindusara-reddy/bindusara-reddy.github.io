//! bindusara-reddy.github.io — the whole site, written in Rust,
//! compiled to WebAssembly. Each page is an empty HTML shell with a
//! data-page attribute; everything you see is rendered from here.

mod engine;
mod nn;
mod pages;
mod util;

use util::document;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let page = document()
        .body()
        .and_then(|b| b.get_attribute("data-page"))
        .unwrap_or_default();
    match page.as_str() {
        "index" => pages::render_index(),
        "gallery" => pages::render_gallery(),
        "playlist" => pages::render_playlist(),
        "note" => pages::render_note(),
        "nn" => nn::render(),
        "404" => pages::render_notfound(),
        _ => {}
    }
}
