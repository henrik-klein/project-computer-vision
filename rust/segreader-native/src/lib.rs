use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView2};

/// Two-pass sequential labeling with Union-Find (4-connectivity).
///
/// Matches the Python reference in `labeling.py` semantically:
///   - background (0) stays 0
///   - foreground pixels get compact consecutive labels starting at 1
///   - 4-connectivity only (north + west neighbors checked in first pass)
///   - labels are not guaranteed to be in the same order as the Python
///     reference, but the partition (which pixels share a label) is identical
fn label_components_rs(binary: ArrayView2<u8>) -> Array2<i32> {
    let (h, w) = binary.dim();
    let mut labels: Array2<i32> = Array2::zeros((h, w));

    if h == 0 || w == 0 {
        return labels;
    }

    // Union-Find with path compression and union by rank.
    // Maximum number of provisional labels = h * w + 1 (label 0 is background).
    let max_labels = h * w + 1;
    let mut parent: Vec<usize> = (0..max_labels).collect();
    let mut rank: Vec<u8> = vec![0u8; max_labels];

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        // Two-pass path compression
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != root {
            let next = parent[x];
            parent[x] = root;
            x = next;
        }
        root
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<u8>, x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return;
        }
        // Union by rank
        if rank[rx] < rank[ry] {
            parent[rx] = ry;
        } else if rank[rx] > rank[ry] {
            parent[ry] = rx;
        } else {
            parent[ry] = rx;
            rank[rx] += 1;
        }
    }

    let mut next_label: usize = 1;

    // Pass 1: assign provisional labels, record equivalences.
    for r in 0..h {
        for c in 0..w {
            if binary[[r, c]] == 0 {
                continue;
            }
            let north = if r > 0 { labels[[r - 1, c]] as usize } else { 0 };
            let west = if c > 0 { labels[[r, c - 1]] as usize } else { 0 };

            let lbl = match (north, west) {
                (0, 0) => {
                    let l = next_label;
                    next_label += 1;
                    l
                }
                (0, w_lbl) => w_lbl,
                (n_lbl, 0) => n_lbl,
                (n_lbl, w_lbl) => {
                    if n_lbl != w_lbl {
                        union(&mut parent, &mut rank, n_lbl, w_lbl);
                    }
                    n_lbl.min(w_lbl)
                }
            };
            labels[[r, c]] = lbl as i32;
        }
    }

    // Pass 2: resolve equivalences, compact to consecutive IDs starting at 1.
    let mut root_to_id: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
    let mut counter: i32 = 1;

    for r in 0..h {
        for c in 0..w {
            let lbl = labels[[r, c]];
            if lbl == 0 {
                continue;
            }
            let root = find(&mut parent, lbl as usize);
            let id = root_to_id.entry(root).or_insert_with(|| {
                let id = counter;
                counter += 1;
                id
            });
            labels[[r, c]] = *id;
        }
    }

    labels
}

/// Connected-component labeling for a 2D binary uint8 array.
///
/// Semantics match the Python reference in `labeling.py`:
///   - input: 2D uint8, 0 = background, nonzero = foreground
///   - output: int32 array, background = 0, foreground labeled 1..N
///   - 4-connectivity (north + west neighbors)
///   - two-pass algorithm with Union-Find (path compression + union by rank)
///   - labels are compact consecutive starting at 1
#[pyfunction]
fn label_components<'py>(
    py: Python<'py>,
    binary: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray2<i32>> {
    let arr = binary.as_array();
    let result = label_components_rs(arr);
    result.into_pyarray(py)
}

/// Smoke test: returns "segreader_native ok".
#[pyfunction]
fn ping() -> &'static str {
    "segreader_native ok"
}

#[pymodule]
fn segreader_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(label_components, m)?)?;
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::label_components_rs;
    use numpy::ndarray::array;

    #[test]
    fn empty_image_returns_zeros() {
        let binary = numpy::ndarray::Array2::<u8>::zeros((0, 0));
        let result = label_components_rs(binary.view());
        assert_eq!(result.dim(), (0, 0));
    }

    #[test]
    fn all_background_stays_zero() {
        let binary = numpy::ndarray::Array2::<u8>::zeros((3, 3));
        let result = label_components_rs(binary.view());
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn single_pixel_gets_label_one() {
        let mut binary = numpy::ndarray::Array2::<u8>::zeros((3, 3));
        binary[[1, 1]] = 1;
        let result = label_components_rs(binary.view());
        assert_eq!(result[[1, 1]], 1);
        assert_eq!(result[[0, 0]], 0);
    }

    #[test]
    fn full_image_is_one_component() {
        let binary = numpy::ndarray::Array2::<u8>::ones((4, 4));
        let result = label_components_rs(binary.view());
        // All pixels must share the same label.
        let first = result[[0, 0]];
        assert!(first > 0);
        assert!(result.iter().all(|&v| v == first));
    }

    #[test]
    fn two_separated_blobs_get_distinct_labels() {
        // Two isolated foreground pixels far apart.
        let mut binary = numpy::ndarray::Array2::<u8>::zeros((1, 5));
        binary[[0, 0]] = 1;
        binary[[0, 4]] = 1;
        let result = label_components_rs(binary.view());
        assert_ne!(result[[0, 0]], 0);
        assert_ne!(result[[0, 4]], 0);
        assert_ne!(result[[0, 0]], result[[0, 4]]);
    }

    #[test]
    fn diagonal_pixels_are_not_connected() {
        // (0,0) and (1,1) only touch diagonally — must be separate components.
        let mut binary = numpy::ndarray::Array2::<u8>::zeros((2, 2));
        binary[[0, 0]] = 1;
        binary[[1, 1]] = 1;
        let result = label_components_rs(binary.view());
        assert_ne!(result[[0, 0]], result[[1, 1]]);
    }

    #[test]
    fn l_shape_is_one_component() {
        // L-shape exercises the union merge path:
        //   1 0
        //   1 0
        //   1 1  ← west neighbor merges top part with the horizontal bar
        let binary = array![
            [1u8, 0],
            [1,   0],
            [1,   1],
        ];
        let result = label_components_rs(binary.view());
        let lbl = result[[0, 0]];
        assert!(lbl > 0);
        assert_eq!(result[[1, 0]], lbl);
        assert_eq!(result[[2, 0]], lbl);
        assert_eq!(result[[2, 1]], lbl);
        assert_eq!(result[[0, 1]], 0);
    }

    #[test]
    fn u_shape_is_one_component() {
        // U-shape — north and west neighbors of the bottom-right corner must
        // both be merged into a single component.
        //   1 0 1
        //   1 0 1
        //   1 1 1
        let binary = array![
            [1u8, 0, 1],
            [1,   0, 1],
            [1,   1, 1],
        ];
        let result = label_components_rs(binary.view());
        let lbl = result[[0, 0]];
        assert!(lbl > 0);
        assert_eq!(result[[0, 2]], lbl);
        assert_eq!(result[[2, 2]], lbl);
        assert_eq!(result[[1, 1]], 0);
    }
}
