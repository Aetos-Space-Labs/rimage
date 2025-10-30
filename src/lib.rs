use pyo3::prelude::*;
use numpy::PyReadonlyArray2;
use numpy::ndarray::{ArrayView, Ix2};

type XY = [i32; 2];
type ImgView<'a> = ArrayView<'a, u16, Ix2>;

type Unit = ();
const UNIT: Unit = ();

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
    max_vals: Vec<u16>,
    max_xy: Vec<XY>,
}

fn find_block_max(iw: &ImgView, x0: i32, x1: i32, y0: i32, y1: i32) -> (u16, XY) {
    let mut best_xy = [x0, y0];
    let mut best_value = 0u16;

    for y in y0..y1 {
        let row = iw.row(y as usize);

        for x in x0..x1 {
            let value = row[x as usize];

            if value > best_value {
                best_value = value;
                best_xy = [x, y];
            }
        }
    }

    (best_value, best_xy)
}

#[pymethods]
impl PyBlockNms {
    fn run(&mut self, data: PyReadonlyArray2<u16>, dist_blocks: i32, total: usize) -> PyResult<Vec<XY>> {
        // Given image or raw radiometry data: divide image into blocks, find per-block MAX, perform NMS
        let mut out: Vec<XY> = Vec::new();
        let iw = data.as_array();

        let start = self.border_skip;
        let end_y = self.blocks_h - start;
        let end_x = self.blocks_w - start;

        for by in start..end_y {
            let y0 = by * self.square;
            let y1 = (y0 + self.square).min(self.height);

            for bx in start..end_x {
                let x0 = bx * self.square;
                let x1 = (x0 + self.square).min(self.width);
                let block_idx = (by * self.blocks_w + bx) as usize;
                let (best_value, best_xy) = find_block_max(&iw, x0, x1, y0, y1);
                self.max_vals[block_idx] = best_value;
                self.max_xy[block_idx] = best_xy;
            }
        }

        self.suppressed.fill(false);

        self.order.sort_by(|&a, &b| {
            let val_a = self.max_vals[a];
            let val_b = self.max_vals[b];
            val_b.cmp(&val_a)
        });

        for &block_idx in &self.order {
            let bx = (block_idx as i32) % self.blocks_w;
            let by = (block_idx as i32) / self.blocks_w;

            if bx < start || bx >= end_x || by < start || by >= end_y {
                continue;
            }

            if self.suppressed[block_idx] || self.max_vals[block_idx] == 0 {
                continue;
            }
            
            let item = self.max_xy[block_idx];
            out.push(item);

            if out.len() == total {
                break;
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

        let max_vals = vec![0; nblocks];
        let max_xy = vec![[0, 0]; nblocks];
        let suppressed = vec![false; nblocks];
        let mut order = Vec::with_capacity(nblocks);

        for i in 0..nblocks {
            order.push(i);
        }

        Self { 
            border_skip, square, height, width, 
            blocks_w, blocks_h, suppressed, order, 
            max_vals, max_xy,
        }
    }
}

#[pymodule]
fn rimage(m: &Bound<'_, PyModule>) -> PyResult<Unit> {
    m.add_class::<PyBlockNms>(/**/)?;
    Ok(UNIT)
}

