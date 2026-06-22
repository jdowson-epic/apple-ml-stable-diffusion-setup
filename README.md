# Apple ml-stable-diffusion on macOS — Setup & Usage

End-to-end guide for installing [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion),
converting a Stable Diffusion model to Core ML format, and running inference on Apple Silicon.

## System requirements

| Requirement | Notes |
|---|---|
| macOS 12+ | Ventura or later recommended |
| Apple Silicon (M1/M2/M3/M4) | Required for Neural Engine (`CPU_AND_NE` / `ALL`) |
| Python **3.12** | 3.13 has breaking numpy/tokenizers incompatibilities (see [Compatibility notes](#compatibility-notes)) |
| ~10 GB free disk space | Model weights (~4 GB) + CoreML packages (~4 GB) |

---

## Automated setup (steps 1–4)

If you'd rather not run the steps manually, `setup.sh` automates them:

```bash
chmod +x setup.sh && ./setup.sh
```

Then skip to [step 5](#5-authenticate-with-huggingface). Otherwise, follow the manual steps below.

---

## 1. Create a Python 3.12 virtual environment

The package's pinned dependencies (numpy < 1.24, transformers == 4.44.2) are incompatible with
Python 3.13. Use Python 3.12 to avoid build failures.

```bash
python3.12 -m venv ~/sd-venv
source ~/sd-venv/bin/activate
pip install --upgrade pip setuptools
```

---

## 2. Install core dependencies

Install `torch`, `coremltools`, and `numpy` first — order matters to avoid
numpy version conflicts introduced by later packages.

```bash
pip install "numpy<2" "torch==2.7.0" coremltools
```

---

## 3. Install apple/ml-stable-diffusion

Clone and install without strict dependency resolution (the package's `numpy<1.24`
constraint is overly conservative and would fail on Python 3.12):

```bash
git clone https://github.com/apple/ml-stable-diffusion.git /tmp/ml-stable-diffusion
pip install --no-deps /tmp/ml-stable-diffusion
rm -rf /tmp/ml-stable-diffusion
```

Then install the remaining runtime dependencies:

```bash
pip install \
  "diffusers[torch]==0.30.2" \
  "transformers==4.44.2" \
  "huggingface-hub==0.24.6" \
  "diffusionkit==0.4.0" \
  pytest invisible-watermark matplotlib safetensors scikit-learn scipy Pillow
```

> **Note:** `diffusionkit` will pull in numpy 2.x as a transitive dependency.
> This is expected — see [Compatibility notes](#compatibility-notes) for the coremltools patch
> required to handle it.

---

## 4. Patch coremltools for numpy 2.x compatibility

coremltools 9.0 has a bug where it calls `int(numpy_array)` in a context that numpy 2.x
no longer allows (implicit array→scalar conversion was removed). Apply a one-line fix:

Find the file:
```bash
python -c "import coremltools, os; print(os.path.dirname(coremltools.__file__))"
# e.g. ~/sd-venv/lib/python3.12/site-packages/coremltools
```

Open `coremltools/converters/mil/frontend/torch/ops.py` and find the `_cast` function
(search for `def _cast`). Change this line:

```python
# Before (line ~3048):
res = mb.const(val=dtype(x.val), name=node.name)

# After:
_val = x.val.item() if hasattr(x.val, 'item') else x.val
res = mb.const(val=dtype(_val), name=node.name)
```

---

## 5. Authenticate with HuggingFace

Most Stable Diffusion models require accepting a license on HuggingFace before downloading.

1. Create a free account at <https://huggingface.co>
2. Accept the model license on the model's page (e.g. [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5))
3. Create an access token at <https://huggingface.co/settings/tokens>
4. Log in:

```bash
huggingface-cli login
# paste your token when prompted
```

---

## 6. Convert a model to Core ML

This converts the three required components (text encoder, UNet, VAE decoder) to
`.mlpackage` format using `SPLIT_EINSUM` attention, which is optimised for the
Apple Neural Engine.

```bash
python -m python_coreml_stable_diffusion.torch2coreml \
  --model-version runwayml/stable-diffusion-v1-5 \
  --convert-text-encoder \
  --convert-vae-decoder \
  --convert-unet \
  --attention-implementation SPLIT_EINSUM \
  --compute-unit ALL \
  -o ~/sd-coreml
```

**Expected output** in `~/sd-coreml/`:

```
Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage
Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_vae_decoder.mlpackage
Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage
```

Conversion takes **20–40 minutes** depending on machine. Model weights (~4 GB) are
downloaded from HuggingFace on first run and cached in `~/.cache/huggingface/`.

### Attention implementation options

| Flag | Best for | Compute unit |
|---|---|---|
| `SPLIT_EINSUM` | Apple Neural Engine | `ALL` or `CPU_AND_NE` |
| `SPLIT_EINSUM_V2` | Newer ANE (M2+) | `ALL` or `CPU_AND_NE` |
| `ORIGINAL` | GPU | `CPU_AND_GPU` |

---

## 7. Run inference

A ready-to-run `generate.py` script is included in this repository:

```bash
# Default prompt, 20 steps, seed 42
python generate.py

# Custom options
python generate.py \
  --prompt "a watercolour painting of a fox in the snow" \
  --steps 30 \
  --seed 7 \
  --output fox.png \
  --compute-unit CPU_AND_NE
```

Full usage:

```
usage: generate.py [-h] [--prompt TEXT] [--steps N] [--seed N] [--output PATH]
                   [--model HF_ID] [--mldir PATH]
                   [--compute-unit {ALL,CPU_AND_NE,CPU_AND_GPU,CPU_ONLY}]
```

> **Note:** The first run JIT-compiles each Core ML model (~2 min total). Compiled
> models are cached by the OS so subsequent runs start in seconds.

---

## Compatibility notes

### Why Python 3.12 (not 3.13)?

| Issue | Root cause |
|---|---|
| `tokenizers 0.19.x` build fails | Uses `PyUnicode_FromKindAndData` — removed in Python 3.13 |
| `numpy < 1.24` build fails | `pkgutil.ImpImporter` — removed in Python 3.12+ |
| Overall | The package's pinned deps predate Python 3.13 |

### Why patch coremltools?

`diffusionkit 0.4.0` (a hard dependency of ml-stable-diffusion 1.1.0) requires numpy 2.x.
coremltools 9.0 was written for numpy 1.x and contains an implicit array→scalar cast
that numpy 2.x rejects. The patch in [step 4](#4-patch-coremltools-for-numpy-2x-compatibility)
fixes this without modifying any other behaviour.

### Models known to work

| Model | Notes |
|---|---|
| `runwayml/stable-diffusion-v1-5` | ✓ Tested — standard UNet, requires HF auth |
| `stabilityai/stable-diffusion-2-1-base` | ✓ Should work — requires HF auth |
| `nota-ai/bk-sdm-small` | ✗ Distilled UNet — unsupported `mid_block_type` |

Models must use a standard `UNetMidBlock2DCrossAttn` architecture.
Distilled or heavily modified UNets are not supported by this package.

---

## Directory layout

```
~/sd-venv/          Python 3.12 virtual environment with all dependencies
~/sd-coreml/        Converted Core ML model packages + scripts
  ├── README.md     This file
  ├── setup.sh      One-shot setup script (automates steps 1–4)
  ├── generate.py   Inference script with CLI flags
  ├── .gitignore
  ├── Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_text_encoder.mlpackage
  ├── Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_unet.mlpackage
  └── Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_vae_decoder.mlpackage
```

## Activating the environment

```bash
source ~/sd-venv/bin/activate
```
