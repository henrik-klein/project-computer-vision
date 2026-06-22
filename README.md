# Seven-Segment Display Reader

A classical computer vision pipeline that reads digits from seven-segment display images — no deep learning, no pretrained models. Every stage is inspectable, parameterizable, and visualizable through a built-in web viewer.

## What it does

Given a photo of a seven-segment display, the pipeline:

1. Optionally localizes the display region using color, brightness, or contour analysis
2. Converts to grayscale and normalizes contrast
3. Binarizes with Otsu's method (or a local-mean adaptive threshold)
4. Cleans the binary image morphologically (opening/closing)
5. Labels connected components using two-pass Union-Find (4-connectivity)
6. Filters components by area and aspect ratio, with fallback logic
7. Separates digit boxes by vertical projection and valley detection
8. Decodes each box by sampling the seven canonical segment zones
9. Annotates and optionally saves debug output for every step

The project intentionally uses classical CV methods so each stage can be inspected, tweaked, and explained — there is no black box.

## Pipeline stages

| # | Stage | Module |
|---|-------|--------|
| 1 | Load and optionally resize image | `io.py` |
| 2 | Display localization (color / brightness / contour / auto) | `localize.py` |
| 3 | Grayscale conversion | `preprocess.py` |
| 4 | CLAHE contrast normalization | `preprocess.py` |
| 5 | Otsu or local-mean binarization | `preprocess.py` |
| 6 | Morphological opening / closing | `morphology.py` |
| 7 | Connected-component labeling (Two-pass Union-Find) | `labeling.py` |
| 8 | Component filtering with fallback | `segment.py` |
| 9 | Vertical projection → digit box detection | `segment.py` |
| 10 | Seven-segment zone sampling and decoding | `decode.py` |
| 11 | Annotated overlay + optional step images | `visualisation.py` |

## Repository structure

```
main.py                      — single-image CLI entry point
evaluate.py                  — batch evaluation with metrics and diagnostics
python/segreader/            — core CV library
  pipeline.py                — run_pipeline() orchestrates all stages
  localize.py                — display region localization
  preprocess.py              — grayscale, contrast, binarization
  morphology.py              — morphological operations
  labeling.py                — Python CCL reference (frozen)
  segment.py                 — component filtering, digit box detection
  decode.py                  — seven-segment zone sampling and decoding
  backend.py                 — backend selector (python / rust)
  steps_io.py                — save pipeline steps as PNG + JSON
  metrics.py                 — accuracy metrics and diagnostics
web_viewer/
  index.html                 — evaluation report viewer (static)
  pipeline_viewer.html       — per-image pipeline step viewer (static)
data/testing/
  images/                    — test images
  samples.json               — ground-truth labels (tracked)
  result.json                — latest evaluation output (git-ignored)
data/processed/              — per-image step PNGs + steps.json (git-ignored)
rust/segreader-native/       — optional Rust CCL backend
  src/lib.rs                 — PyO3 extension: label_components()
  README.md                  — Rust build instructions
tests/                       — pytest test suite
```

## Setup

```powershell
uv sync
uv run pytest
```

## Run one image

```powershell
uv run python main.py data/samples/pic1.png --backend python --no-save
```

Expected output:

```
142530
```

Available options:

```
-b / --backend      python (default) | rust
-t / --threshold    Segment fill threshold 0.0–1.0 (default 0.40)
-n / --no-save      Skip saving the annotated output image
-v / --verbose      Print each digit individually
-p / --processed    Save all pipeline step images to data/processed/<stem>/
--no-localize       Skip display localization, use full image
```

## Batch evaluation

Run the full test set and produce a result JSON for the web viewer:

```powershell
uv run python evaluate.py data/testing/images -w 2 `
    -o data/testing/result_python.json `
    --ground-truth data/testing/samples.json `
    --diagnostics --backend python --viewer-result
```

`--viewer-result` writes an additional copy to `data/testing/result.json` so the web viewer picks it up automatically — no manual file copying needed.

Available sweep modes (add `--ground-truth` to use):

```
--sweep-thresholds  T1,T2,...    — accuracy at different segment thresholds
--sweep-morphology               — compare morphology kernel variants
--sweep-binarization             — compare binarization methods
--sweep-localization             — compare localization strategies
```

## Web viewer

Start a static HTTP server from the project root:

```powershell
python -m http.server 8000
```

Open: `http://localhost:8000/web_viewer/index.html`

The viewer fetches `/data/testing/samples.json` and `/data/testing/result.json` automatically. After running `evaluate.py --viewer-result` the page reflects the latest results without any manual copying.

Click any image card to open a details modal. If processed steps exist for that image, the modal loads the pipeline viewer showing every CV stage with its intermediate image and metadata.

### Generate pipeline step images

```powershell
uv run python main.py "data/testing/images/image copy 10.png" `
    --backend python --no-save --processed
```

Steps are written to `data/processed/image copy 10/`. The viewer links to them automatically from the card modal.

## Optional Rust backend

The Rust backend implements connected-component labeling using the same two-pass Union-Find algorithm as the Python reference. It is optional — the project runs without it.

### Build

See `rust/README.md` for full prerequisites (MinGW-w64 via scoop, stable GNU toolchain).

```powershell
cd rust/segreader-native
cargo test --target x86_64-pc-windows-gnu
cd ../..

$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
uv run maturin develop `
    --manifest-path rust/segreader-native/Cargo.toml `
    --target x86_64-pc-windows-gnu
```

### Verify

```powershell
uv run python -c "import segreader_native; print(segreader_native.ping())"
uv run python main.py data/samples/pic1.png --backend rust --no-save
```

Expected:

```
segreader_native ok
142530
```

### Evaluate with Rust backend

```powershell
uv run python evaluate.py data/testing/images -w 2 `
    -o data/testing/result_rust.json `
    --ground-truth data/testing/samples.json `
    --diagnostics --backend rust --viewer-result
```

## Current accuracy and limitations

| Metric | Value |
|--------|-------|
| Labeled test images | 26 |
| Correct (default settings) | ~8 / 26 |
| Accuracy (labeled) | ~31% |

The main bottleneck is binarization on difficult images: reversed-LCD displays, strong glare, or very low contrast. Localization, morphology, threshold, and digit-box detection have each been swept over multiple variants — none closed the gap significantly. The project prioritizes explainability and diagnostic completeness over black-box accuracy improvement.

## Tests

```powershell
uv run pytest
```

If Rust is installed:

```powershell
cd rust/segreader-native
cargo test --target x86_64-pc-windows-gnu
```

## Demo checklist

```powershell
# 1. Verify all tests pass
uv run pytest

# 2. Run one sample image
uv run python main.py data/samples/pic1.png --backend python --no-save

# 3. Run batch evaluation and update viewer
uv run python evaluate.py data/testing/images -w 2 `
    -o data/testing/result_python.json `
    --ground-truth data/testing/samples.json `
    --diagnostics --backend python --viewer-result

# 4. Start the web server and open the viewer
python -m http.server 8000
# → http://localhost:8000/web_viewer/index.html
```

## Notes

- `data/testing/samples.json` is the ground-truth label file and is tracked in git.
- `data/testing/result*.json` and `data/processed/` are git-ignored (generated).
- The web viewer is fully static — it reads JSON files via HTTP but executes no server-side code.
- The Python backend is always the default; `--backend rust` requires the compiled extension.
