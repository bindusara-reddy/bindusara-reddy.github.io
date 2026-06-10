//! Every content page of the site, rendered from Rust.

use crate::util::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

struct Card {
    title: &'static str,
    preview: &'static str,
    image: &'static str,
    link: &'static str,
}

fn card_html(c: &Card) -> String {
    let preview = if c.preview.is_empty() {
        String::new()
    } else {
        format!("<p>{}</p>", c.preview)
    };
    format!(
        r##"<div class="item">
            <img src="{img}" alt="{title}" loading="lazy">
            <div class="item-content">
                <h3>{title}</h3>
                {preview}
                <a href="{link}">Don't Click</a>
            </div>
        </div>"##,
        img = c.image,
        title = c.title,
        link = c.link,
        preview = preview
    )
}

fn section_html(id: &str, cards: &[Card]) -> String {
    let inner: String = cards.iter().map(card_html).collect();
    format!(
        r##"<section id="{id}" class="content" aria-label="{id}">
            <h2 class="section-label">{id}</h2>{inner}
        </section>"##
    )
}

pub fn render_index() {
    let note = [Card {
        title: "H-ll-W-rld",
        preview: "Do you like videogames?",
        image: "https://i.pinimg.com/736x/d4/09/b2/d409b254a9ad71f0225993123fea6840.jpg",
        link: "Note/helloworld.html",
    }];
    let clip = [Card {
        title: "catto",
        preview: "",
        image: "https://i.pinimg.com/564x/c4/93/20/c49320b072068b5da9b176d1a77adce7.jpg",
        link: "Clip/gallery.html",
    }];
    let memory = [Card {
        title: "SoundWaves",
        preview: "Vibes..",
        image: "https://i.pinimg.com/564x/11/92/33/1192331815ef0cc6e9934d5c87aba5f3.jpg",
        link: "Memory/playlist.html",
    }];
    let thought = [Card {
        title: "neuralnet",
        preview: "How a computer reads numbers:",
        image: "https://i.pinimg.com/564x/01/18/d1/0118d1ecf9211898da08a8225079c33b.jpg",
        link: "Thought/intelligence.html",
    }];

    let body = format!(
        r##"<header>
        <h1>Bindusara Reddy</h1>
        <a class="scroll-hint" href="#note" aria-label="Scroll down to content">&#8595;</a>
    </header>
    <nav>
        <a href="#note">Note</a>
        <a href="#clip">Clip</a>
        <a href="#memory">Memory</a>
        <a href="#thought">Thought</a>
    </nav>
    <div class="container">{s1}{s2}{s3}{s4}</div>
    <footer>
        <p>&copy; Journey to the stars🚀🌟<a href="poem.html" style="text-decoration: none;">❤️</a></p>
    </footer>"##,
        s1 = section_html("note", &note),
        s2 = section_html("clip", &clip),
        s3 = section_html("memory", &memory),
        s4 = section_html("thought", &thought),
    );
    document().body().unwrap().set_inner_html(&body);
    reveal_items();
}

fn reveal_items() {
    if reduced_motion() {
        return;
    }
    let cb = Closure::wrap(Box::new(
        move |entries: js_sys::Array, obs: web_sys::IntersectionObserver| {
            for e in entries.iter() {
                let entry: web_sys::IntersectionObserverEntry = e.dyn_into().unwrap();
                if entry.is_intersecting() {
                    let el: web_sys::Element = entry.target();
                    let _ = el.class_list().add_1("visible");
                    obs.unobserve(&el);
                }
            }
        },
    )
        as Box<dyn FnMut(js_sys::Array, web_sys::IntersectionObserver)>);
    if let Ok(obs) = web_sys::IntersectionObserver::new(cb.as_ref().unchecked_ref()) {
        let items = document().query_selector_all(".item").unwrap();
        for i in 0..items.length() {
            if let Some(node) = items.item(i) {
                let el: web_sys::Element = node.dyn_into().unwrap();
                let _ = el.class_list().add_1("reveal");
                obs.observe(&el);
            }
        }
    }
    cb.forget();
}

pub fn render_gallery() {
    let body = r##"<a class="home" href="../index.html">&larr; home</a>
    <h1>Gallery</h1>
    <div class="gallery">
        <img src="catto/1.jpg" alt="catto 1" loading="lazy">
        <img src="catto/2.jpg" alt="catto 2" loading="lazy">
        <img src="catto/3.jpg" alt="catto 3" loading="lazy">
    </div>
    <div class="lightbox" id="lightbox" role="dialog" aria-label="Enlarged photo"><img src="" alt=""></div>"##;
    document().body().unwrap().set_inner_html(body);

    let imgs = document().query_selector_all(".gallery img").unwrap();
    for i in 0..imgs.length() {
        let el: web_sys::Element = imgs.item(i).unwrap().dyn_into().unwrap();
        let el2 = el.clone();
        on(el.as_ref(), "click", move |_| {
            let lb = get("lightbox");
            let img = lb.query_selector("img").unwrap().unwrap();
            let _ = img.set_attribute("src", &el2.get_attribute("src").unwrap_or_default());
            let _ = img.set_attribute("alt", &el2.get_attribute("alt").unwrap_or_default());
            let _ = lb.class_list().add_1("open");
        });
    }
    on(get("lightbox").as_ref(), "click", |_| {
        let _ = get("lightbox").class_list().remove_1("open");
    });
    on(document().as_ref(), "keydown", |e| {
        let ke: web_sys::KeyboardEvent = e.dyn_into().unwrap();
        if ke.key() == "Escape" {
            let _ = get("lightbox").class_list().remove_1("open");
        }
    });
}

pub fn render_playlist() {
    let body = r##"<a class="home" href="../index.html">&larr; home</a>
    <h1>Music</h1>
    <div class="playlist">
        <h2>YouTube Videos</h2>
        <ul>
            <li><a href="https://www.youtube.com/watch?v=hQ5x8pHoIPA" target="_blank" rel="noopener">Feather</a></li>
            <li><a href="https://www.youtube.com/watch?v=L53gjP-TtGE" target="_blank" rel="noopener">Power</a></li>
            <li><a href="https://www.youtube.com/watch?v=J3DWAJGaf7o" target="_blank" rel="noopener">Lost</a></li>
        </ul>
    </div>
    <div class="playlist">
        <h2>Spotify Songs</h2>
        <ul>
            <li><a href="https://open.spotify.com/track/3GVkPk8mqxz0itaAriG1L7" target="_blank" rel="noopener">Everybody dies in their nightmares</a></li>
            <li><a href="https://open.spotify.com/track/315aBOUD3xtj7sUMXtRgMV" target="_blank" rel="noopener">In the Stars</a></li>
            <li><a href="https://open.spotify.com/track/4nVBt6MZDDP6tRVdQTgxJg" target="_blank" rel="noopener">Story of my life</a></li>
        </ul>
    </div>"##;
    document().body().unwrap().set_inner_html(body);
}

pub fn render_note() {
    let body = r##"<nav>
        <a href="../index.html">Home</a>
        <a href="../index.html#note">Note</a>
        <a href="../index.html#clip">Clip</a>
        <a href="../index.html#memory">Memory</a>
        <a href="../index.html#thought">Thought</a>
    </nav>
    <div class="post-content">
        <h1>Hello World</h1>
        <p class="post-date">23 June 2024</p>
        <img src="https://i.pinimg.com/564x/2a/9c/80/2a9c8079b3e75e2d60df0bdd7d4793cd.jpg" alt="Elden Ring" loading="lazy">
        <p>Dark Souls:</p>
        <p>A journey against the entropic decay of the universe</p>
    </div>
    <footer>
        <p>&copy; Journey to the stars🚀🌟❤️</p>
    </footer>"##;
    document().body().unwrap().set_inner_html(body);
}

pub fn render_notfound() {
    let body = r##"<div class="drift">🚀</div>
    <h1>404</h1>
    <p>you drifted off the map</p>
    <a href="/">back to earth</a>"##;
    document().body().unwrap().set_inner_html(body);
}
