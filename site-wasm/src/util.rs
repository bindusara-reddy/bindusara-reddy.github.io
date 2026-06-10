use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, Document, Element, EventTarget, HtmlCanvasElement, HtmlElement,
    HtmlInputElement, Storage, Window,
};

pub fn window() -> Window {
    web_sys::window().expect("no window")
}

pub fn document() -> Document {
    window().document().expect("no document")
}

pub fn get(id: &str) -> Element {
    document()
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("missing #{id}"))
}

pub fn get_html(id: &str) -> HtmlElement {
    get(id).dyn_into().unwrap()
}

pub fn get_input(id: &str) -> HtmlInputElement {
    get(id).dyn_into().unwrap()
}

pub fn get_canvas(id: &str) -> HtmlCanvasElement {
    get(id).dyn_into().unwrap()
}

pub fn ctx2d(canvas: &HtmlCanvasElement) -> CanvasRenderingContext2d {
    canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap()
}

pub fn set_text(id: &str, s: &str) {
    get(id).set_text_content(Some(s));
}

// version-proof style setters (the typed web-sys setters have churned)
pub fn fill_style(ctx: &CanvasRenderingContext2d, s: &str) {
    let _ = js_sys::Reflect::set(ctx.as_ref(), &"fillStyle".into(), &s.into());
}

pub fn stroke_style(ctx: &CanvasRenderingContext2d, s: &str) {
    let _ = js_sys::Reflect::set(ctx.as_ref(), &"strokeStyle".into(), &s.into());
}

pub fn on<F: FnMut(web_sys::Event) + 'static>(target: &EventTarget, ev: &str, f: F) {
    let c = Closure::wrap(Box::new(f) as Box<dyn FnMut(web_sys::Event)>);
    target
        .add_event_listener_with_callback(ev, c.as_ref().unchecked_ref())
        .unwrap();
    c.forget();
}

pub fn on_click<F: FnMut() + 'static>(id: &str, mut f: F) {
    on(get(id).as_ref(), "click", move |_| f());
}

// requestAnimationFrame loop; the callback gets the timestamp and
// returns false to stop the loop
pub fn raf_loop<F: FnMut(f64) -> bool + 'static>(f: F) {
    use std::cell::RefCell;
    use std::rc::Rc;
    let f = Rc::new(RefCell::new(f));
    let cb: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let cb2 = cb.clone();
    *cb.borrow_mut() = Some(Closure::wrap(Box::new(move |t: f64| {
        if (f.borrow_mut())(t) {
            if let Some(c) = cb2.borrow().as_ref() {
                let _ = window().request_animation_frame(c.as_ref().unchecked_ref());
            }
        } else {
            cb2.borrow_mut().take();
        }
    }) as Box<dyn FnMut(f64)>));
    let _ = window().request_animation_frame(cb.borrow().as_ref().unwrap().as_ref().unchecked_ref());
}

pub fn reduced_motion() -> bool {
    window()
        .match_media("(prefers-reduced-motion: reduce)")
        .ok()
        .flatten()
        .map(|m| m.matches())
        .unwrap_or(false)
}

pub fn storage() -> Option<Storage> {
    window().local_storage().ok().flatten()
}

pub fn now() -> f64 {
    window().performance().map(|p| p.now()).unwrap_or(0.0)
}

pub fn random() -> f64 {
    js_sys::Math::random()
}

pub fn ease_out(t: f64) -> f64 {
    1.0 - (1.0 - t).powi(3)
}

pub fn clamp01(t: f64) -> f64 {
    t.clamp(0.0, 1.0)
}
