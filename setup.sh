#!/usr/bin/env zsh
# setup.sh — Automates README steps 1–4 for apple/ml-stable-diffusion on macOS
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [--venv-dir PATH]   (default venv: ~/sd-venv)
#
# After this script completes:
#   source ~/sd-venv/bin/activate          # activate the environment
#   huggingface-cli login                   # step 5: authenticate (one-time)
#   python -m python_coreml_stable_diffusion.torch2coreml ...  # step 6: convert
#   python generate.py                      # step 7: run inference

set -euo pipefail

VENV_DIR="${1:-$HOME/sd-venv}"
PYTHON="python3.12"

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "==> Checking prerequisites..."

if ! command -v "$PYTHON" &>/dev/null; then
  echo "ERROR: $PYTHON not found. Install Python 3.12 via Homebrew or pyenv:"
  echo "  brew install python@3.12"
  exit 1
fi

PYTHON_VER=$("$PYTHON" --version 2>&1)
echo "    Python: $PYTHON_VER"
echo "    Venv  : $VENV_DIR"
echo

# ── Step 1: Create virtual environment ────────────────────────────────────────
echo "==> [Step 1] Creating Python 3.12 virtual environment at $VENV_DIR..."
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools -q --progress-bar off
echo "    Done."
echo

# ── Step 2: Install core dependencies ─────────────────────────────────────────
echo "==> [Step 2] Installing numpy<2, torch==2.7.0, coremltools..."
echo "    (This downloads ~2 GB — may take a few minutes)"
"$VENV_DIR/bin/pip" install "numpy<2" "torch==2.7.0" coremltools \
  --progress-bar off -q
echo "    Done."
echo

# ── Step 3: Install apple/ml-stable-diffusion ─────────────────────────────────
echo "==> [Step 3] Installing apple/ml-stable-diffusion and runtime dependencies..."

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "    Cloning repository..."
git clone --depth=1 https://github.com/apple/ml-stable-diffusion.git "$TMPDIR/ml-sd" -q

echo "    Installing package (--no-deps)..."
"$VENV_DIR/bin/pip" install --no-deps "$TMPDIR/ml-sd" --progress-bar off -q

echo "    Installing runtime dependencies..."
"$VENV_DIR/bin/pip" install \
  "diffusers[torch]==0.30.2" \
  "transformers==4.44.2" \
  "huggingface-hub==0.24.6" \
  "diffusionkit==0.4.0" \
  pytest invisible-watermark matplotlib safetensors scikit-learn scipy Pillow \
  --progress-bar off -q
echo "    Done."
echo

# ── Step 4: Patch coremltools for numpy 2.x ───────────────────────────────────
echo "==> [Step 4] Patching coremltools for numpy 2.x compatibility..."

OPS_PY=$("$VENV_DIR/bin/python" \
  -c "import coremltools, os; print(os.path.dirname(coremltools.__file__))" \
  2>/dev/null)/converters/mil/frontend/torch/ops.py

if grep -q "res = mb.const(val=dtype(x.val)" "$OPS_PY" 2>/dev/null; then
  sed -i '' \
    's/            res = mb\.const(val=dtype(x\.val), name=node\.name)/            _val = x.val.item() if hasattr(x.val, '"'"'item'"'"') else x.val\n            res = mb.const(val=dtype(_val), name=node.name)/' \
    "$OPS_PY"
  echo "    Patch applied to $OPS_PY"
elif grep -q "_val = x.val.item()" "$OPS_PY" 2>/dev/null; then
  echo "    Patch already applied — skipping."
else
  echo "    WARNING: Could not locate patch target in $OPS_PY"
  echo "    Apply the patch manually (see README step 4)."
fi
echo

# ── Done ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    source $VENV_DIR/bin/activate"
echo "    huggingface-cli login          # paste your HF token"
echo ""
echo "  Then convert a model (step 6 in README), and generate:"
echo "    python generate.py --prompt 'your prompt here'"
echo "============================================================"
