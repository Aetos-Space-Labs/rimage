use pyo3::prelude::*;
use numpy::PyReadonlyArray2;
use numpy::ndarray::{ArrayView, Ix2};

type Unit = (/**/);
type PyInfoVec = Vec<PyBlockInfo>;
type ImgView<'a> = ArrayView<'a, u16, Ix2>;
const UNIT: Unit = (/**/);

#[pyclass]
struct PyBlockNms {
    border_skip: i32, 
    square: i32,
    height: i32,
    width: i32,

    blocks_w: i32,
    blocks_h: i32,

    order: Vec<usize>,
    suppressed: Vec<bool>,
    maxes: PyInfoVec,
}

#[pyclass]
#[derive(Clone)]
struct PyBlockInfo {
    #[pyo3(get)] max: u16,
    #[pyo3(get)] psr: f64,
    #[pyo3(get)] x: i32,
    #[pyo3(get)] y: i32,
}

impl Default for PyBlockInfo {
    fn default(/**/) -> Self {
        Self { max: 0, psr: 0f64, x: 0, y: 0 }
    }
}

fn get_block_info(iw: &ImgView, x0: i32, x1: i32, y0: i32, y1: i32) -> PyBlockInfo {
    let mut info = PyBlockInfo { max: 0u16, psr: 0f64, x: x0, y: y0 };

    for y in y0..y1 {
        let row = iw.row(y as usize);

        for x in x0..x1 {
            let value = row[x as usize];

            if value > info.max {
                info.max = value;
                info.x = x;
                info.y = y;
            }
        }
    }

    info
}

#[pymethods]
impl PyBlockNms {
    fn run(&mut self, data: PyReadonlyArray2<u16>, dist_blocks: i32, total: usize) -> PyResult<PyInfoVec> {
        // Given image or raw radiometry data: divide image into blocks, find per-block MAX, perform NMS
        let mut out: PyInfoVec = Vec::new(/**/);
        let iw = data.as_array(/**/);

        let start = self.border_skip;
        let end_y = self.blocks_h - start;
        let end_x = self.blocks_w - start;

        let mut nblocks = 0f64;
        let mut sum = 0f64;
        let mut sq = 0f64;

        for by in start..end_y {
            let y0 = by * self.square;
            let y1 = (y0 + self.square).min(self.height);

            for bx in start..end_x {
                let x0 = bx * self.square;
                let x1 = (x0 + self.square).min(self.width);

                let info = get_block_info(&iw, x0, x1, y0, y1);
                let block_idx = (by * self.blocks_w + bx) as usize;
                let best_safe_value = info.max as f64;

                self.maxes[block_idx] = info;
                sq += best_safe_value.powi(2);
                sum += best_safe_value;
                nblocks += 1f64;
            }
        }

        let mean = sum / nblocks;
        let variance = sq / nblocks - mean.powi(2);
        let std = variance.sqrt(/**/).max(0.00001);

        self.suppressed.fill(false);

        self.order.sort_by(|&a, &b| {
            let val_a = self.maxes[a].max;
            let val_b = self.maxes[b].max;
            val_b.cmp(&val_a)
        });

        for &block_idx in &self.order {
            if self.suppressed[block_idx] || self.maxes[block_idx].max == 0 {
                continue
            }

            let bx = (block_idx as i32) % self.blocks_w;
            let by = (block_idx as i32) / self.blocks_w;

            if bx < start || bx >= end_x || by < start || by >= end_y {
                continue
            }
            
            let mut info = self.maxes[block_idx].clone(/**/);
            info.psr = (info.max as f64 - mean) / std;
            out.push(info);

            if out.len(/**/) == total {
                break
            }

            let y_min = (by - dist_blocks).max(start);
            let x_min = (bx - dist_blocks).max(start);
            let y_max = (by + dist_blocks).min(end_y - 1);
            let x_max = (bx + dist_blocks).min(end_x - 1);

            for yy in y_min..=y_max {
                for xx in x_min..=x_max {
                    let nb = yy * self.blocks_w + xx;
                    self.suppressed[nb as usize] = true;
                }
            }
        }

        Ok(out)
    }

    #[new]
    fn new(border_skip: i32, square: i32, height: i32, width: i32) -> Self {
        let blocks_h = (height + square - 1) / square;
        let blocks_w = (width + square - 1) / square;
        let nblocks = (blocks_h * blocks_w) as usize;

        let suppressed = vec![false; nblocks];
        let mut order = Vec::with_capacity(nblocks);
        let maxes = vec![PyBlockInfo::default(/**/); nblocks];

        for i in 0..nblocks {
            order.push(i);
        }

        Self { 
            border_skip, square, height, width, 
            blocks_w, blocks_h, suppressed, 
            order, maxes,
        }
    }
}

#[pymodule]
fn rimage(m: &Bound<'_, PyModule>) -> PyResult<Unit> {
    m.add_class::<PyBlockInfo>(/**/)?;
    m.add_class::<PyBlockNms>(/**/)?;
    Ok(UNIT)
}

