//! A simple program to be proven inside the zkVM.

#![no_main]
sp1_zkvm::entrypoint!(main);

// copied from https://github.com/LaurentMazare/mamba.rs/blob/main/src/constants.rs
pub mod params_130m {
    // https://huggingface.co/state-spaces/mamba-130m/blob/main/config.json
    pub const D_MODEL: usize = 768;
    pub const N_LAYER: usize = 24;
    pub const MODEL_FILENAME: &str = "mamba-130m.bin";
}

pub mod params_370m {
    // https://huggingface.co/state-spaces/mamba-370m/blob/main/config.json
    pub const D_MODEL: usize = 1024;
    pub const N_LAYER: usize = 48;
    pub const MODEL_FILENAME: &str = "mamba-370m.bin";
}

pub mod params_790m {
    // https://huggingface.co/state-spaces/mamba-790m/blob/main/config.json
    pub const D_MODEL: usize = 1536;
    pub const N_LAYER: usize = 48;
    pub const MODEL_FILENAME: &str = "mamba-790m.bin";
}

pub mod params_1_4b {
    // https://huggingface.co/state-spaces/mamba-1.4b/blob/main/config.json
    pub const D_MODEL: usize = 2048;
    pub const N_LAYER: usize = 48;
    pub const MODEL_FILENAME: &str = "mamba-1.4b.bin";
}

pub mod params_2_8b {
    // https://huggingface.co/state-spaces/mamba-2.8b/blob/main/config.json
    pub const D_MODEL: usize = 2560;
    pub const N_LAYER: usize = 64;
    pub const MODEL_FILENAME: &str = "mamba-2.8b.bin";
}

pub const VOCAB_SIZE_: usize = 50277;
pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 8;
pub const VOCAB_SIZE: usize =
    (VOCAB_SIZE_ + PAD_VOCAB_SIZE_MULTIPLE - 1) / PAD_VOCAB_SIZE_MULTIPLE * PAD_VOCAB_SIZE_MULTIPLE;

pub const D_CONV: usize = 4;
pub const D_STATE: usize = 16;







// copied from https://github.com/LaurentMazare/mamba.rs/blob/main/src/model.rs
// #![allow(clippy::needless_range_loop)]
use rayon::prelude::*;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LinearNoBias<const IN: usize, const OUT: usize> {
    w: [[f32; IN]; OUT],
}

impl<const IN: usize, const OUT: usize> LinearNoBias<IN, OUT> {
    // https://github.com/srush/llama2.rs/blob/2ca8f3dc0d4aa945a29700271883af72d9043ef1/src/model.rs#L22
    pub fn forward<const B: usize>(&self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) {
        for (xout, x) in xout.iter_mut().zip(x) {
            xout.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = self.w[i].iter().zip(x.iter()).fold(0.0, |acc, (&_w, &_x)| acc + _w * _x);
            });
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct RmsNorm<const DIM: usize> {
    w: [f32; DIM],
}

impl<const DIM: usize> RmsNorm<DIM> {
    fn forward_one(&self, o: &mut [f32; DIM], xo: &[f32; DIM], epsilon: f32) {
        // calculate sum of squares
        let mut ss = xo.iter().fold(0.0, |acc, x| acc + x * x);

        // take mean
        ss /= DIM as f32;
        ss += epsilon;
        ss = 1.0 / ss.sqrt();
        // normalize and scale
        for (j, weight_j) in self.w.iter().enumerate() {
            // Solve some borrow nonsense.
            o[j] = weight_j * ss * xo[j];
        }
    }

    fn forward<const B: usize>(
        &self,
        outs: &mut [[f32; DIM]; B],
        ins: &[[f32; DIM]; B],
        epsilon: f32,
    ) {
        for (outs, ins) in outs.iter_mut().zip(ins.iter()) {
            self.forward_one(outs, ins, epsilon);
        }
    }
}

fn add_in_place(a: &mut [f32], b: &[f32]) {
    for (a_i, b_i) in a.iter_mut().zip(b) {
        *a_i += b_i;
    }
}

fn mul_in_place(a: &mut [f32], b: &[f32]) {
    for (a_i, b_i) in a.iter_mut().zip(b) {
        *a_i *= b_i;
    }
}

fn silu_in_place(s: &mut [f32]) {
    for s in s.iter_mut() {
        *s = *s * (1.0 / (1.0 + (-*s).exp()));
    }
}

fn softplus_in_place(s: &mut [f32]) {
    // No softplus threshold here...
    for s in s.iter_mut() {
        *s = (s.exp() + 1.).ln()
    }
}

fn dot<const B: usize>(v1: &[f32; B], v2: &[f32; B]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&v1, &v2)| v1 * v2).sum::<f32>()
}

pub trait ModelWeights {
    type State<const B: usize>;
    const MODEL_FILENAME: &'static str;

    fn new_state<const B: usize>() -> Self::State<B>;
    fn update_state<const B: usize>(&self, state: &mut Self::State<B>, tokens: &[u32; B]);
    fn state_logits<const B: usize>(state: &Self::State<B>) -> &[[f32; VOCAB_SIZE]; B];
}

// pub mod model_130m {
//     use super::*;
//     pub use params_130m::*;
//     include!("model_inc.rs");
// }

// pub mod model_370m {
//     use super::*;
//     pub use params_370m::*;
//     include!("model_inc.rs");
// }

// pub mod model_790m {
//     use super::*;
//     pub use params_790m::*;
//     include!("model_inc.rs");
// }

// pub mod model_1_4b {
//     use super::*;
//     pub use params_1_4b::*;
//     include!("model_inc.rs");
// }

// pub mod model_2_8b {
//     use super::*;
//     pub use params_2_8b::*;
//     include!("model_inc.rs");
// }






// copied from https://github.com/LaurentMazare/mamba.rs/blob/main/src/model_inc.rs

// This file is included multiple times so as to use the different value for the constants.
// Once the adt_const_params or generic_const_exprs features are available, this could be
// removed.
use params_130m::D_MODEL;
use params_130m::N_LAYER;
use params_130m::MODEL_FILENAME;

pub const D_INNER: usize = D_MODEL * 2;
pub const DT_RANK: usize = (D_MODEL + 15) / 16;

// The file is mmaped hence enforcing the C representation.
#[repr(C)]
struct BlockWeights {
    norm: RmsNorm<D_MODEL>,
    in_proj1: LinearNoBias<D_MODEL, D_INNER>,
    in_proj2: LinearNoBias<D_MODEL, D_INNER>,
    x_proj1: LinearNoBias<D_INNER, DT_RANK>,
    x_proj2: LinearNoBias<D_INNER, D_STATE>,
    x_proj3: LinearNoBias<D_INNER, D_STATE>,
    dt_proj: LinearNoBias<DT_RANK, D_INNER>,
    dt_proj_bias: [f32; D_INNER],
    out_proj: LinearNoBias<D_INNER, D_MODEL>,
    a: [[f32; D_STATE]; D_INNER],
    d: [f32; D_INNER],
    conv1d_weight: [[f32; D_INNER]; D_CONV],
    conv1d_bias: [f32; D_INNER],
}

#[repr(C)]
pub struct Weights {
    embedding: [[f32; D_MODEL]; VOCAB_SIZE],
    layers: [BlockWeights; N_LAYER],
    norm_f: RmsNorm<D_MODEL>,
    lm_head: LinearNoBias<D_MODEL, VOCAB_SIZE>,
}

pub struct State<const B: usize> {
    // Persistent state
    hs: [[[[f32; D_STATE]; D_INNER]; B]; N_LAYER],
    prev_xs: [[[[f32; D_INNER]; B]; D_CONV]; N_LAYER],
    pos: usize,

    // Temporary variables, pre-allocated and only used in [update]
    xs: [[f32; D_MODEL]; B],
    norm_xs: [[f32; D_MODEL]; B],
    logits: [[f32; VOCAB_SIZE]; B],
    delta: [[f32; DT_RANK]; B],
    delta_proj: [[f32; D_INNER]; B],
    b: [[f32; D_STATE]; B],
    c: [[f32; D_STATE]; B],
    proj_for_conv: [[f32; D_INNER]; B],
    proj_for_silu: [[f32; D_INNER]; B],
}

impl<const B: usize> State<B> {
    pub fn new() -> Self {
        Self {
            hs: [[[[0f32; D_STATE]; D_INNER]; B]; N_LAYER],
            prev_xs: [[[[0f32; D_INNER]; B]; D_CONV]; N_LAYER],
            pos: 0,

            xs: [[0f32; D_MODEL]; B],
            norm_xs: [[0f32; D_MODEL]; B],
            logits: [[0f32; VOCAB_SIZE]; B],
            delta: [[0f32; DT_RANK]; B],
            delta_proj: [[0f32; D_INNER]; B],
            b: [[0f32; D_STATE]; B],
            c: [[0f32; D_STATE]; B],
            proj_for_conv: [[0f32; D_INNER]; B],
            proj_for_silu: [[0f32; D_INNER]; B],
        }
    }

    pub fn update(&mut self, tokens: &[u32; B], w: &Weights) {
        for (xs, token) in self.xs.iter_mut().zip(tokens) {
            xs.copy_from_slice(&w.embedding[*token as usize]);
        }

        // See Figure 3, page 8, on https://arxiv.org/pdf/2312.00752.pdf
        for ((layer, hs), prev_xs) in
            w.layers.iter().zip(self.hs.iter_mut()).zip(self.prev_xs.iter_mut())
        {
            layer.norm.forward(&mut self.norm_xs, &self.xs, 1e-5);

            {
                // Mixer forward.
                layer.in_proj1.forward(&mut self.proj_for_conv, &self.norm_xs);
                layer.in_proj2.forward(&mut self.proj_for_silu, &self.norm_xs);

                let pos = self.pos % D_STATE;
                for b in 0..B {
                    prev_xs[pos % D_CONV][b].copy_from_slice(&self.proj_for_conv[b])
                }
                // Apply the conv1d and put the result in proj_for_conv.
                for (b, proj_for_conv) in self.proj_for_conv.iter_mut().enumerate() {
                    proj_for_conv.copy_from_slice(&layer.conv1d_bias);
                    for d_c in 0..D_CONV {
                        for d_i in 0..D_INNER {
                            proj_for_conv[d_i] += layer.conv1d_weight[d_c][d_i]
                                * prev_xs[(d_c + 1 + pos) % D_CONV][b][d_i]
                        }
                    }
                }

                for s in self.proj_for_conv.iter_mut() {
                    silu_in_place(s)
                }
                {
                    // SSM + Selection, we're doing inference here so only need the last step of
                    // the sequence.
                    // Algorithm 3.2 on page 6, https://arxiv.org/pdf/2312.00752.pdf
                    layer.x_proj1.forward(&mut self.delta, &self.proj_for_conv);
                    layer.x_proj2.forward(&mut self.b, &self.proj_for_conv);
                    layer.x_proj3.forward(&mut self.c, &self.proj_for_conv);

                    // Weird, what isn't this multiplication combined with x_proj1?
                    layer.dt_proj.forward(&mut self.delta_proj, &self.delta);
                    for delta_proj in self.delta_proj.iter_mut() {
                        add_in_place(delta_proj, &layer.dt_proj_bias)
                    }
                    for s in self.delta_proj.iter_mut() {
                        softplus_in_place(s);
                    }

                    // Selective scan part
                    for b in 0..B {
                        // Eqn (2a), page 3, h_t = Ab h_{t-1} + Bb x_t
                        for d_i in 0..D_INNER {
                            let delta = self.delta_proj[b][d_i];
                            let x = self.proj_for_conv[b][d_i];
                            for d_s in 0..D_STATE {
                                let a = layer.a[d_i][d_s];
                                let b_ = self.b[b][d_s];
                                hs[b][d_i][d_s] =
                                    hs[b][d_i][d_s] * (delta * a).exp() + delta * b_ * x;
                            }
                        }
                    }
                    // Put the result back in proj_for_conv
                    // y_t = c * h_t
                    for b in 0..B {
                        for d_i in 0..D_INNER {
                            self.proj_for_conv[b][d_i] = dot(&self.c[b], &hs[b][d_i])
                                + layer.d[d_i] * self.proj_for_conv[b][d_i]
                        }
                    }
                }

                for (s_out, s_in) in self.proj_for_silu.iter_mut().zip(self.proj_for_conv.iter()) {
                    silu_in_place(s_out);
                    mul_in_place(s_out, s_in);
                }
                // Put the result back in norm_xs
                layer.out_proj.forward(&mut self.norm_xs, &self.proj_for_silu)
            }

            // Residual connections
            for (norm_xs, xs) in self.norm_xs.iter().zip(self.xs.iter_mut()) {
                add_in_place(xs, norm_xs)
            }
        }
        self.pos += 1;

        w.norm_f.forward(&mut self.norm_xs, &self.xs, 1e-5);
        w.lm_head.forward(&mut self.logits, &self.norm_xs)
    }

    pub fn logits(&self) -> &[[f32; VOCAB_SIZE]; B] {
        &self.logits
    }
}

impl ModelWeights for Weights {
    type State<const B: usize> = State<B>;
    const MODEL_FILENAME: &'static str = MODEL_FILENAME;

    fn new_state<const B: usize>() -> Self::State<B> {
        Self::State::new()
    }

    fn update_state<const B: usize>(&self, state: &mut Self::State<B>, tokens: &[u32; B]) {
        state.update(tokens, self)
    }

    fn state_logits<const B: usize>(state: &Self::State<B>) -> &[[f32; VOCAB_SIZE]; B] {
        state.logits()
    }
}

impl<const B: usize> Default for State<B> {
    fn default() -> Self {
        Self::new()
    }
}







// copied from https://github.com/LaurentMazare/mamba.rs/blob/main/src/tokenizer.rs

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

const BYTES_TO_UNICODE: [(u8, char); 256] = [
    (33, '!'),
    (34, '"'),
    (35, '#'),
    (36, '$'),
    (37, '%'),
    (38, '&'),
    (39, '\''),
    (40, '('),
    (41, ')'),
    (42, '*'),
    (43, '+'),
    (44, ','),
    (45, '-'),
    (46, '.'),
    (47, '/'),
    (48, '0'),
    (49, '1'),
    (50, '2'),
    (51, '3'),
    (52, '4'),
    (53, '5'),
    (54, '6'),
    (55, '7'),
    (56, '8'),
    (57, '9'),
    (58, ':'),
    (59, ';'),
    (60, '<'),
    (61, '='),
    (62, '>'),
    (63, '?'),
    (64, '@'),
    (65, 'A'),
    (66, 'B'),
    (67, 'C'),
    (68, 'D'),
    (69, 'E'),
    (70, 'F'),
    (71, 'G'),
    (72, 'H'),
    (73, 'I'),
    (74, 'J'),
    (75, 'K'),
    (76, 'L'),
    (77, 'M'),
    (78, 'N'),
    (79, 'O'),
    (80, 'P'),
    (81, 'Q'),
    (82, 'R'),
    (83, 'S'),
    (84, 'T'),
    (85, 'U'),
    (86, 'V'),
    (87, 'W'),
    (88, 'X'),
    (89, 'Y'),
    (90, 'Z'),
    (91, '['),
    (92, '\\'),
    (93, ']'),
    (94, '^'),
    (95, '_'),
    (96, '`'),
    (97, 'a'),
    (98, 'b'),
    (99, 'c'),
    (100, 'd'),
    (101, 'e'),
    (102, 'f'),
    (103, 'g'),
    (104, 'h'),
    (105, 'i'),
    (106, 'j'),
    (107, 'k'),
    (108, 'l'),
    (109, 'm'),
    (110, 'n'),
    (111, 'o'),
    (112, 'p'),
    (113, 'q'),
    (114, 'r'),
    (115, 's'),
    (116, 't'),
    (117, 'u'),
    (118, 'v'),
    (119, 'w'),
    (120, 'x'),
    (121, 'y'),
    (122, 'z'),
    (123, '{'),
    (124, '|'),
    (125, '}'),
    (126, '~'),
    (161, '¡'),
    (162, '¢'),
    (163, '£'),
    (164, '¤'),
    (165, '¥'),
    (166, '¦'),
    (167, '§'),
    (168, '¨'),
    (169, '©'),
    (170, 'ª'),
    (171, '«'),
    (172, '¬'),
    (174, '®'),
    (175, '¯'),
    (176, '°'),
    (177, '±'),
    (178, '²'),
    (179, '³'),
    (180, '´'),
    (181, 'µ'),
    (182, '¶'),
    (183, '·'),
    (184, '¸'),
    (185, '¹'),
    (186, 'º'),
    (187, '»'),
    (188, '¼'),
    (189, '½'),
    (190, '¾'),
    (191, '¿'),
    (192, 'À'),
    (193, 'Á'),
    (194, 'Â'),
    (195, 'Ã'),
    (196, 'Ä'),
    (197, 'Å'),
    (198, 'Æ'),
    (199, 'Ç'),
    (200, 'È'),
    (201, 'É'),
    (202, 'Ê'),
    (203, 'Ë'),
    (204, 'Ì'),
    (205, 'Í'),
    (206, 'Î'),
    (207, 'Ï'),
    (208, 'Ð'),
    (209, 'Ñ'),
    (210, 'Ò'),
    (211, 'Ó'),
    (212, 'Ô'),
    (213, 'Õ'),
    (214, 'Ö'),
    (215, '×'),
    (216, 'Ø'),
    (217, 'Ù'),
    (218, 'Ú'),
    (219, 'Û'),
    (220, 'Ü'),
    (221, 'Ý'),
    (222, 'Þ'),
    (223, 'ß'),
    (224, 'à'),
    (225, 'á'),
    (226, 'â'),
    (227, 'ã'),
    (228, 'ä'),
    (229, 'å'),
    (230, 'æ'),
    (231, 'ç'),
    (232, 'è'),
    (233, 'é'),
    (234, 'ê'),
    (235, 'ë'),
    (236, 'ì'),
    (237, 'í'),
    (238, 'î'),
    (239, 'ï'),
    (240, 'ð'),
    (241, 'ñ'),
    (242, 'ò'),
    (243, 'ó'),
    (244, 'ô'),
    (245, 'õ'),
    (246, 'ö'),
    (247, '÷'),
    (248, 'ø'),
    (249, 'ù'),
    (250, 'ú'),
    (251, 'û'),
    (252, 'ü'),
    (253, 'ý'),
    (254, 'þ'),
    (255, 'ÿ'),
    (0, 'Ā'),
    (1, 'ā'),
    (2, 'Ă'),
    (3, 'ă'),
    (4, 'Ą'),
    (5, 'ą'),
    (6, 'Ć'),
    (7, 'ć'),
    (8, 'Ĉ'),
    (9, 'ĉ'),
    (10, 'Ċ'),
    (11, 'ċ'),
    (12, 'Č'),
    (13, 'č'),
    (14, 'Ď'),
    (15, 'ď'),
    (16, 'Đ'),
    (17, 'đ'),
    (18, 'Ē'),
    (19, 'ē'),
    (20, 'Ĕ'),
    (21, 'ĕ'),
    (22, 'Ė'),
    (23, 'ė'),
    (24, 'Ę'),
    (25, 'ę'),
    (26, 'Ě'),
    (27, 'ě'),
    (28, 'Ĝ'),
    (29, 'ĝ'),
    (30, 'Ğ'),
    (31, 'ğ'),
    (32, 'Ġ'),
    (127, 'ġ'),
    (128, 'Ģ'),
    (129, 'ģ'),
    (130, 'Ĥ'),
    (131, 'ĥ'),
    (132, 'Ħ'),
    (133, 'ħ'),
    (134, 'Ĩ'),
    (135, 'ĩ'),
    (136, 'Ī'),
    (137, 'ī'),
    (138, 'Ĭ'),
    (139, 'ĭ'),
    (140, 'Į'),
    (141, 'į'),
    (142, 'İ'),
    (143, 'ı'),
    (144, 'Ĳ'),
    (145, 'ĳ'),
    (146, 'Ĵ'),
    (147, 'ĵ'),
    (148, 'Ķ'),
    (149, 'ķ'),
    (150, 'ĸ'),
    (151, 'Ĺ'),
    (152, 'ĺ'),
    (153, 'Ļ'),
    (154, 'ļ'),
    (155, 'Ľ'),
    (156, 'ľ'),
    (157, 'Ŀ'),
    (158, 'ŀ'),
    (159, 'Ł'),
    (160, 'ł'),
    (173, 'Ń'),
];

const PAT: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    re: fancy_regex::Regex,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, String>,
    encoder: HashMap<String, u32>,
    decoder: Vec<String>,
    bpe_ranks: HashMap<(Vec<u8>, Vec<u8>), u32>,
}

impl Tokenizer {
    /// Creates a new tokenizer, this takes as input the path for the bpe rank file.
    pub fn new<T: AsRef<std::path::Path>>(vocab_path: T, merge_path: T) -> Result<Tokenizer> {
        let re = fancy_regex::Regex::new(PAT)?;
        let mut byte_encoder = [' '; 256];
        let mut byte_decoder = HashMap::new();
        for &(byte, unicode) in BYTES_TO_UNICODE.iter() {
            byte_decoder.insert(unicode, String::from_utf8_lossy(&[byte]).to_string());
            byte_encoder[byte as usize] = unicode
        }
        let encoder = std::fs::read_to_string(vocab_path)?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&encoder)?;
        let mut decoder = Vec::new();
        for (token_str, token_id) in encoder.iter() {
            let token_id = *token_id as usize;
            if token_id >= decoder.len() {
                decoder.resize(token_id + 1, "".to_string())
            }
            decoder[token_id] = token_str.clone()
        }
        let merge_file = std::fs::File::open(merge_path)?;
        let merge_file = std::io::BufReader::new(merge_file);
        let mut bpe_ranks = HashMap::new();
        for line in merge_file.lines() {
            let line = line?;
            let line = line.split(' ').collect::<Vec<_>>();
            if line.len() == 2 {
                let key = (line[0].as_bytes().to_vec(), line[1].as_bytes().to_vec());
                bpe_ranks.insert(key, bpe_ranks.len() as u32);
            }
        }
        Ok(Tokenizer { re, byte_decoder, byte_encoder, encoder, decoder, bpe_ranks })
    }

    /// The main tokenization entry point, takes as input a string and returns the list of tokens.
    pub fn encode(&self, s: &str) -> anyhow::Result<Vec<u32>> {
        let mut bpe_tokens: Vec<u32> = vec![];
        for word in self.re.find_iter(s) {
            let word = word?;
            let mut encoded_word = vec![];
            for &byte in word.as_str().as_bytes() {
                encoded_word.push(self.byte_encoder[byte as usize])
            }
            let encoded_word: String = encoded_word.iter().collect();
            bpe_tokens.extend(self.bpe(&encoded_word))
        }
        Ok(bpe_tokens)
    }

    fn get_pairs(word: &[Vec<u8>]) -> HashSet<(Vec<u8>, Vec<u8>)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, word: &str) -> Vec<u32> {
        let mut word: Vec<Vec<u8>> = word.chars().map(|x| x.to_string().into_bytes()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        while word.len() > 1 {
            let mut current_min = None;
            let pairs = Self::get_pairs(&word);
            for p in pairs.iter() {
                match self.bpe_ranks.get(p) {
                    None => {}
                    Some(v) => {
                        let should_replace = match current_min {
                            None => true,
                            Some((current_min, _)) => v < current_min,
                        };
                        if should_replace {
                            current_min = Some((v, p))
                        }
                    }
                }
            }
            let (first, second) = match current_min {
                None => break,
                Some((_v, (first, second))) => (first, second),
            };
            let mut new_word = vec![];
            let mut index = 0;
            while index < word.len() {
                let w = &word[index];
                if index + 1 < word.len() && w == first && &word[index + 1] == second {
                    let mut first_and_second = first.clone();
                    first_and_second.extend_from_slice(second);
                    new_word.push(first_and_second);
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter()
            .filter_map(|x| {
                let x = String::from_utf8_lossy(x).to_string();
                self.encoder.get(&x)
            })
            .copied()
            .collect()
    }

    /// The inverse of the tokenization process, takes as input a list of tokens and returns a
    /// string that produces this tokenization.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut str = vec![];
        for &token in tokens.iter() {
            let token = token as usize;
            if token >= self.decoder.len() {
                anyhow::bail!("token {token} is out of range {}", self.decoder.len())
            }
            str.push(self.decoder[token].as_str())
        }
        Ok(str.concat())
    }

    pub fn get_token(&self, s: &str) -> Option<u32> {
        self.encoder.get(s).copied()
    }

    // This should be memoized if it starts to be on the critical path.
    pub fn decode_token_id(&self, token_id: u32) -> Result<String> {
        let token_id = token_id as usize;
        if token_id >= self.decoder.len() {
            anyhow::bail!("token {token_id} is out of range {}", self.decoder.len())
        }
        let mut chars = vec![];
        for c in self.decoder[token_id].chars() {
            let c = match self.byte_decoder.get(&c) {
                None => c.to_string(),
                Some(s) => s.to_string(),
            };
            chars.push(c)
        }
        Ok(chars.concat())
    }
}


// #![feature(portable_simd)]
use clap::{Parser, ValueEnum};
use rand::{distributions::Distribution, SeedableRng};
use std::io::Write;

// This struct is self-referential in a sense as if mmap gets dropped, weights would not be valid
// anymore.
struct MmapedWeights<W: ModelWeights + 'static> {
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    weights: &'static W,
}

impl<W: ModelWeights> MmapedWeights<W> {
    /// This function is unsafe as it uses mmap and doesn't check the file size.
    fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        println!("loading weights from path {p:?}");
        let file = std::fs::File::open(p)
            .map_err(|e| anyhow::Error::new(e).context(format!("trying to read {p:?}")))?;
        let file_len = file.metadata()?.len();
        println!("=====================================================");
        println!("file length: {file_len} bytes");
        println!("=====================================================");
        let expected_len = std::mem::size_of::<W>() as u64;
        if file_len != expected_len {
            anyhow::bail!("Unexpected length of file for {p:?}, {file_len} <> {expected_len}")
        }
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // the dodgy bit.
        let weights = unsafe { &*(mmap.as_ptr() as *const W) };
        Ok(Self { mmap, weights })
    }

    fn weights(&self) -> &W {
        self.weights
    }
}


fn main() {
    let prompt_tokens = &sp1_zkvm::io::read::<Vec<u32>>();
    let temperature = 0.0;
    run::<Weights>(prompt_tokens, temperature)
    // match args.which {
    //     Which::M130m => run::<model_130m::Weights>(args.prompt, args.temperature),
    //     Which::M370m => run::<model_370m::Weights>(args.prompt, args.temperature),
    //     Which::M790m => run::<model::model_790m::Weights>(args.prompt, args.temperature),
    //     Which::M1_4b => run::<model::model_1_4b::Weights>(args.prompt, args.temperature),
    //     Which::M2_8b => run::<model::model_2_8b::Weights>(args.prompt, args.temperature),
    // }
}

fn run<W: ModelWeights + 'static>(prompt_tokens: &Vec<u32>, temperature: f64) {
    let mut state = Box::new(W::new_state::<1>());
    let tokenizer = Tokenizer::new("vocab.json", "merges.txt").expect("cannot load tokenizer");
    let mmaped_weights: MmapedWeights<W> = MmapedWeights::from_file(W::MODEL_FILENAME).expect("cannot load weights");
    println!("state size:  {:4}MB", std::mem::size_of::<W::State<1>>() >> 20);
    println!("weight size: {:4}MB", std::mem::size_of::<W>() >> 20);
    let mut lp = LogitsProcessor::new(299792458, temperature);
    let eos_token = match tokenizer.get_token("<|endoftext|>") {
        Some(token) => token,
        None => panic!("cannot find the </s> token"),
    };
    // println!("processing prompt '{prompt}'");
    // let prompt_tokens = tokenizer.encode(&prompt).unwrap();
    // println!("prompt tokens: {prompt_tokens:?}");

    for &token_id in prompt_tokens.iter() {
        mmaped_weights.weights().update_state(&mut state, &[token_id]);
        let token_str = tokenizer.decode_token_id(token_id).expect("cannot decode token");
        print!("{token_str}")
    }
    // std::io::stdout().flush();

    // let start_gen = std::time::Instant::now();
    // let mut generated_tokens = 0usize;
    let next_token = lp.sample(&W::state_logits(&state)[0]).expect("cannot sample next token");
    println!("{next_token}");
    sp1_zkvm::io::write(&next_token);
    // loop {
    //     let next_token = lp.sample(&W::state_logits(&state)[0])?;
    //     if next_token == eos_token {
    //         println!();
    //         break;
    //     }
    //     let next_token_str = tokenizer.decode_token_id(next_token)?;
    //     print!("{next_token_str}");
    //     mmaped_weights.weights().update_state(&mut state, &[next_token]);
    //     generated_tokens += 1;
    // }
    // let dt = start_gen.elapsed();
    // println!(
    //     "\n{generated_tokens} tokens generated ({:.2} token/s)",
    //     generated_tokens as f64 / dt.as_secs_f64(),
    // );
    // Ok(())
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: f64) -> Self {
        let temperature = if temperature < 1e-7 { None } else { Some(temperature) };
        Self { rng: rand::rngs::StdRng::seed_from_u64(seed), temperature }
    }

    fn sample_argmax(&mut self, logits: &[f32]) -> Result<u32> {
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    pub fn sample(&mut self, logits: &[f32; VOCAB_SIZE]) -> Result<u32> {
        let next_token = match self.temperature {
            None => self.sample_argmax(logits)?,
            Some(temperature) => {
                let max_logit = logits.iter().max_by(|f1, f2| f1.total_cmp(f2)).unwrap();
                let mut prs = [0f32; VOCAB_SIZE];
                let mut sum_pr = 0f32;
                for (pr, logit) in prs.iter_mut().zip(logits.iter()) {
                    *pr = ((logit - max_logit) / temperature as f32).exp();
                    sum_pr += *pr;
                }
                for pr in prs.iter_mut() {
                    *pr /= sum_pr
                }
                self.sample_multinomial(&prs)?
            }
        };
        Ok(next_token)
    }
}
