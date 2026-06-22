# segreader-native

Native Rust extension providing an optional faster backend for connected-component labeling (CCL).

## What it provides

`segreader_native.label_components(binary)` — two-pass sequential labeling with Union-Find (path compression + union by rank, 4-connectivity). Output is semantically equivalent to the Python reference in `python/segreader/labeling.py`.

The Python default backend remains `"python"`. Select `"rust"` only after building this extension.

## Prerequisites

- [scoop](https://scoop.sh) (user-level package manager, no admin required)
- MinGW-w64 via scoop (provides `as.exe` and `dlltool.exe`)
- Rust stable GNU toolchain

```powershell
# Install scoop
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# Install MinGW-w64
scoop install mingw

# Install Rust (if not yet installed)
# Download rustup-init.exe from https://win.rustup.rs/x86_64 and run with --profile minimal
# Then add the GNU toolchain:
rustup target add x86_64-pc-windows-gnu
rustup toolchain install stable-x86_64-pc-windows-gnu
```

## Build and install

From the project root:

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\mingw\current\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
uv run maturin develop `
    --manifest-path rust/segreader-native/Cargo.toml `
    --target x86_64-pc-windows-gnu
```

## Verify

```python
import segreader_native
print(segreader_native.ping())  # → "segreader_native ok"

import numpy as np
binary = np.array([[1,0],[1,1]], dtype=np.uint8)
print(segreader_native.label_components(binary))
```

## Run Rust unit tests

```powershell
$env:PATH = "$env:USERPROFILE\scoop\apps\mingw\current\bin;$env:USERPROFILE\.cargo\bin;" +
            "C:\Users\cgmar\AppData\Roaming\uv\python\cpython-3.9.25-windows-x86_64-none;$env:PATH"
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
cd rust/segreader-native
cargo test --target x86_64-pc-windows-gnu
```

## Manual smoke test (CLI)

Run both backends on the same image and compare:

```powershell
uv run python main.py data/samples/pic1.png --backend python --no-save
uv run python main.py data/samples/pic1.png --backend rust   --no-save
```

Both commands should print the same number. If `segreader_native` is not installed, the `--backend rust` command exits with code 3 and a clear error message.

## Use from Python pipeline

```python
from segreader import run_pipeline
number, _ = run_pipeline("data/samples/pic1.png", backend="rust")
```

If `segreader_native` is not installed and `backend="rust"` is requested, a clear `ImportError` is raised with build instructions.
