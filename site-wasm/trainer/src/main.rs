// Trains a 784 -> 64 (ReLU) -> 10 (softmax) MLP on MNIST with plain SGD,
// then exports int8-quantized weights as a JS file for the website demo.
use std::fs;

const IN: usize = 784;
const HID: usize = 64;
const OUT: usize = 10;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

fn be32(b: &[u8], o: usize) -> usize {
    ((b[o] as usize) << 24) | ((b[o + 1] as usize) << 16) | ((b[o + 2] as usize) << 8) | (b[o + 3] as usize)
}

fn read_images(path: &str) -> Vec<f32> {
    let b = fs::read(path).expect(path);
    assert_eq!(be32(&b, 0), 2051);
    let n = be32(&b, 4);
    b[16..16 + n * IN].iter().map(|&p| p as f32 / 255.0).collect()
}

fn read_labels(path: &str) -> Vec<u8> {
    let b = fs::read(path).expect(path);
    assert_eq!(be32(&b, 0), 2049);
    let n = be32(&b, 4);
    b[8..8 + n].to_vec()
}

fn forward(w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32], x: &[f32], a1: &mut [f32], p: &mut [f32]) {
    for h in 0..HID {
        let row = &w1[h * IN..(h + 1) * IN];
        let mut s = b1[h];
        for i in 0..IN {
            s += row[i] * x[i];
        }
        a1[h] = if s > 0.0 { s } else { 0.0 };
    }
    let mut mx = f32::NEG_INFINITY;
    for o in 0..OUT {
        let row = &w2[o * HID..(o + 1) * HID];
        let mut s = b2[o];
        for h in 0..HID {
            s += row[h] * a1[h];
        }
        p[o] = s;
        if s > mx {
            mx = s;
        }
    }
    let mut sum = 0.0;
    for o in 0..OUT {
        p[o] = (p[o] - mx).exp();
        sum += p[o];
    }
    for o in 0..OUT {
        p[o] /= sum;
    }
}

fn accuracy(w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32], imgs: &[f32], labels: &[u8]) -> f32 {
    let n = labels.len();
    let mut a1 = vec![0.0; HID];
    let mut p = vec![0.0; OUT];
    let mut ok = 0;
    for s in 0..n {
        forward(w1, b1, w2, b2, &imgs[s * IN..(s + 1) * IN], &mut a1, &mut p);
        let pred = (0..OUT).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()).unwrap();
        if pred == labels[s] as usize {
            ok += 1;
        }
    }
    ok as f32 / n as f32
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 4 / 3 + 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let v = (b0 << 16) | (b1 << 8) | b2;
        s.push(B64[(v >> 18) as usize & 63] as char);
        s.push(B64[(v >> 12) as usize & 63] as char);
        s.push(if chunk.len() > 1 { B64[(v >> 6) as usize & 63] as char } else { '=' });
        s.push(if chunk.len() > 2 { B64[v as usize & 63] as char } else { '=' });
    }
    s
}

fn quantize(w: &[f32]) -> (Vec<u8>, f32) {
    let mx = w.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = mx / 127.0;
    let q: Vec<u8> = w.iter().map(|&v| ((v / scale).round() as i8) as u8).collect();
    (q, scale)
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let out_path = std::env::args().nth(2).unwrap_or_else(|| "weights.js".into());
    let imgs = read_images(&format!("{dir}/train-images-idx3-ubyte"));
    let labels = read_labels(&format!("{dir}/train-labels-idx1-ubyte"));
    let test_imgs = read_images(&format!("{dir}/t10k-images-idx3-ubyte"));
    let test_labels = read_labels(&format!("{dir}/t10k-labels-idx1-ubyte"));
    let n = labels.len();
    println!("train {n}, test {}", test_labels.len());

    let mut rng = Rng(42);
    let mut w1 = vec![0.0f32; HID * IN];
    let mut b1 = vec![0.0f32; HID];
    let mut w2 = vec![0.0f32; OUT * HID];
    let mut b2 = vec![0.0f32; OUT];
    let s1 = (2.0 / IN as f32).sqrt();
    let s2 = (2.0 / HID as f32).sqrt();
    for v in w1.iter_mut() {
        *v = (rng.unit() * 2.0 - 1.0) * s1;
    }
    for v in w2.iter_mut() {
        *v = (rng.unit() * 2.0 - 1.0) * s2;
    }

    let mut order: Vec<usize> = (0..n).collect();
    let mut a1 = vec![0.0f32; HID];
    let mut p = vec![0.0f32; OUT];
    let mut z1pos = vec![false; HID];

    for epoch in 0..14 {
        let lr = 0.05 * 0.85f32.powi(epoch);
        for i in (1..n).rev() {
            let j = (rng.next() as usize) % (i + 1);
            order.swap(i, j);
        }
        for &s in order.iter() {
            let x = &imgs[s * IN..(s + 1) * IN];
            let y = labels[s] as usize;
            forward(&w1, &b1, &w2, &b2, x, &mut a1, &mut p);
            for h in 0..HID {
                z1pos[h] = a1[h] > 0.0;
            }
            // dL/dz2 = p - onehot(y)
            let mut dz2 = p.clone();
            dz2[y] -= 1.0;
            // hidden grad
            let mut dz1 = vec![0.0f32; HID];
            for o in 0..OUT {
                let row = &w2[o * HID..(o + 1) * HID];
                for h in 0..HID {
                    dz1[h] += row[h] * dz2[o];
                }
            }
            for h in 0..HID {
                if !z1pos[h] {
                    dz1[h] = 0.0;
                }
            }
            // updates
            for o in 0..OUT {
                let row = &mut w2[o * HID..(o + 1) * HID];
                let g = dz2[o] * lr;
                for h in 0..HID {
                    row[h] -= g * a1[h];
                }
                b2[o] -= g;
            }
            for h in 0..HID {
                if dz1[h] == 0.0 {
                    continue;
                }
                let row = &mut w1[h * IN..(h + 1) * IN];
                let g = dz1[h] * lr;
                for i in 0..IN {
                    row[i] -= g * x[i];
                }
                b1[h] -= g;
            }
        }
        let acc = accuracy(&w1, &b1, &w2, &b2, &test_imgs, &test_labels);
        println!("epoch {epoch}: lr {lr:.4}, test accuracy {:.2}%", acc * 100.0);
    }

    let final_acc = accuracy(&w1, &b1, &w2, &b2, &test_imgs, &test_labels);
    let (q1, sc1) = quantize(&w1);
    let (q2, sc2) = quantize(&w2);
    let b1s: Vec<String> = b1.iter().map(|v| format!("{v:.5}")).collect();
    let b2s: Vec<String> = b2.iter().map(|v| format!("{v:.5}")).collect();
    let js = format!(
        "// 784->{HID}->10 MLP trained on MNIST (test accuracy {:.2}%)\nconst MNIST_WEIGHTS = {{\n  hid: {HID},\n  acc: {:.2},\n  s1: {sc1:e},\n  s2: {sc2:e},\n  w1: \"{}\",\n  w2: \"{}\",\n  b1: [{}],\n  b2: [{}]\n}};\n",
        final_acc * 100.0,
        final_acc * 100.0,
        base64(&q1),
        base64(&q2),
        b1s.join(","),
        b2s.join(",")
    );
    fs::write(&out_path, js).unwrap();
    println!("wrote {out_path}");

    // binary export for embedding straight into a wasm binary:
    // [hid u32][s1 f32][s2 f32][acc f32][w1 i8][b1 f32][w2 i8][b2 f32] LE
    let mut bin: Vec<u8> = Vec::new();
    bin.extend((HID as u32).to_le_bytes());
    bin.extend(sc1.to_le_bytes());
    bin.extend(sc2.to_le_bytes());
    bin.extend((final_acc * 100.0).to_le_bytes());
    bin.extend(&q1);
    for v in &b1 {
        bin.extend(v.to_le_bytes());
    }
    bin.extend(&q2);
    for v in &b2 {
        bin.extend(v.to_le_bytes());
    }
    let bin_path = format!("{}.bin", out_path.trim_end_matches(".js"));
    fs::write(&bin_path, &bin).unwrap();
    println!("wrote {bin_path} ({} bytes)", bin.len());
}
