//! The neural network itself: a 784 -> 64 -> 10 MLP whose trained weights
//! are compiled into this wasm binary. Forward pass, backprop fine-tuning,
//! and per-browser persistence — all in Rust.

use crate::util::{storage, window};

pub const IN: usize = 784;
pub const OUT: usize = 10;

static WEIGHTS_BIN: &[u8] = include_bytes!("../weights.bin");

const STORE_KEY: &str = "nn-digit-weights-v1";
const COUNT_KEY: &str = "nn-digit-taught-v1";

pub struct Net {
    pub hid: usize,
    pub acc: f32,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Net {
    pub fn factory() -> Net {
        let b = WEIGHTS_BIN;
        let hid = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        let s1 = f32_at(b, 4);
        let s2 = f32_at(b, 8);
        let acc = f32_at(b, 12);
        let mut o = 16;
        let w1: Vec<f32> = b[o..o + hid * IN].iter().map(|&v| v as i8 as f32 * s1).collect();
        o += hid * IN;
        let b1: Vec<f32> = (0..hid).map(|i| f32_at(b, o + i * 4)).collect();
        o += hid * 4;
        let w2: Vec<f32> = b[o..o + OUT * hid].iter().map(|&v| v as i8 as f32 * s2).collect();
        o += OUT * hid;
        let b2: Vec<f32> = (0..OUT).map(|i| f32_at(b, o + i * 4)).collect();
        Net { hid, acc, w1, b1, w2, b2 }
    }

    /// factory weights, then any brain the visitor trained earlier
    pub fn load() -> (Net, bool) {
        let mut net = Net::factory();
        let restored = net.restore();
        (net, restored)
    }

    pub fn w1(&self) -> &[f32] {
        &self.w1
    }

    pub fn w2(&self) -> &[f32] {
        &self.w2
    }

    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, [f32; OUT]) {
        let hid = self.hid;
        let mut a1 = vec![0.0f32; hid];
        for h in 0..hid {
            let row = &self.w1[h * IN..(h + 1) * IN];
            let mut s = self.b1[h];
            for i in 0..IN {
                s += row[i] * x[i];
            }
            a1[h] = if s > 0.0 { s } else { 0.0 };
        }
        let mut z2 = [0.0f32; OUT];
        let mut mx = f32::NEG_INFINITY;
        for o in 0..OUT {
            let row = &self.w2[o * hid..(o + 1) * hid];
            let mut s = self.b2[o];
            for h in 0..hid {
                s += row[h] * a1[h];
            }
            z2[o] = s;
            if s > mx {
                mx = s;
            }
        }
        let mut probs = [0.0f32; OUT];
        let mut sum = 0.0;
        for o in 0..OUT {
            probs[o] = (z2[o] - mx).exp();
            sum += probs[o];
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        (a1, probs)
    }

    /// one lesson: real backpropagation on a single example
    pub fn train(&mut self, x: &[f32], label: usize, lr: f32, reps: usize) -> f32 {
        let hid = self.hid;
        let mut loss = 0.0;
        for _ in 0..reps {
            let (a1, probs) = self.forward(x);
            loss = -(probs[label].max(1e-9)).ln();
            let mut dz2 = probs;
            dz2[label] -= 1.0;
            let mut dz1 = vec![0.0f32; hid];
            for o in 0..OUT {
                let row = &self.w2[o * hid..(o + 1) * hid];
                for h in 0..hid {
                    dz1[h] += row[h] * dz2[o];
                }
            }
            for h in 0..hid {
                if a1[h] <= 0.0 {
                    dz1[h] = 0.0;
                }
            }
            for o in 0..OUT {
                let g = dz2[o] * lr;
                let row = &mut self.w2[o * hid..(o + 1) * hid];
                for h in 0..hid {
                    row[h] -= g * a1[h];
                }
                self.b2[o] -= g;
            }
            for h in 0..hid {
                if dz1[h] == 0.0 {
                    continue;
                }
                let g = dz1[h] * lr;
                let row = &mut self.w1[h * IN..(h + 1) * IN];
                for i in 0..IN {
                    row[i] -= g * x[i];
                }
                self.b1[h] -= g;
            }
        }
        self.save();
        loss
    }

    /// same on-disk format as the old JS engine, so brains taught before
    /// the rust rewrite still load
    fn serialize(&self) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for v in self.w1.iter().chain(&self.b1).chain(&self.w2).chain(&self.b2) {
            bytes.extend(v.to_le_bytes());
        }
        let bin: String = bytes.iter().map(|&b| b as char).collect();
        window().btoa(&bin).unwrap_or_default()
    }

    pub fn save(&self) {
        if let Some(st) = storage() {
            let _ = st.set_item(STORE_KEY, &self.serialize());
        }
    }

    fn restore(&mut self) -> bool {
        let Some(st) = storage() else { return false };
        let Ok(Some(s)) = st.get_item(STORE_KEY) else { return false };
        let Ok(bin) = window().atob(&s) else { return false };
        let total = self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len();
        let bytes: Vec<u8> = bin.chars().map(|c| c as u32 as u8).collect();
        if bytes.len() != total * 4 {
            return false;
        }
        let mut vals = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        for v in self.w1.iter_mut().chain(&mut self.b1).chain(&mut self.w2).chain(&mut self.b2) {
            *v = vals.next().unwrap();
        }
        true
    }

    pub fn reset(&mut self) {
        *self = Net::factory();
        if let Some(st) = storage() {
            let _ = st.remove_item(STORE_KEY);
            let _ = st.remove_item(COUNT_KEY);
        }
    }
}

pub fn taught_count() -> u32 {
    storage()
        .and_then(|s| s.get_item(COUNT_KEY).ok().flatten())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

pub fn bump_taught() -> u32 {
    let n = taught_count() + 1;
    if let Some(st) = storage() {
        let _ = st.set_item(COUNT_KEY, &n.to_string());
    }
    n
}
