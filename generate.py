#!/usr/bin/env python3
"""
CoreML Stable Diffusion image generation script.

Usage:
    python generate.py [--prompt "..."] [--steps N] [--seed N] [--output path.png]

Requires:
    - ~/sd-venv activated (or run via ~/sd-venv/bin/python generate.py)
    - Converted .mlpackage files in MLPACKAGES_DIR (see README step 6)
    - HuggingFace authentication (see README step 5)
"""
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from diffusers import StableDiffusionPipeline
from python_coreml_stable_diffusion import pipeline as coreml_pipeline


# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_VERSION  = "runwayml/stable-diffusion-v1-5"
MLPACKAGES_DIR = str(Path(__file__).parent)   # same directory as this script
COMPUTE_UNIT   = "ALL"                         # CPU + GPU + Neural Engine


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with CoreML Stable Diffusion")
    parser.add_argument("--prompt",  default="a photo of an astronaut riding a horse on the moon",
                        help="Text prompt for image generation")
    parser.add_argument("--steps",   type=int, default=20,
                        help="Number of denoising steps (default: 20)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output",  default="output.png",
                        help="Output image path (default: output.png)")
    parser.add_argument("--model",   default=MODEL_VERSION,
                        help=f"HuggingFace model ID (default: {MODEL_VERSION})")
    parser.add_argument("--mldir",   default=MLPACKAGES_DIR,
                        help=f"Directory containing .mlpackage files (default: {MLPACKAGES_DIR})")
    parser.add_argument("--compute-unit", default=COMPUTE_UNIT,
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
                        help=f"Core ML compute unit (default: {COMPUTE_UNIT})")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'Prompt : "{args.prompt}"')
    print(f"Steps  : {args.steps}   Seed: {args.seed}   Compute: {args.compute_unit}")
    print()

    # 1. Load PyTorch pipeline for tokenizer / scheduler config only
    print("Loading reference pipeline...")
    pytorch_pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,   # not converted to CoreML
    )

    # 2. Swap in CoreML models
    # NOTE: First run JIT-compiles the models (~2 min). Subsequent runs skip this.
    print(f"Loading CoreML models from {args.mldir}...")
    t0 = time.time()
    coreml_pipe = coreml_pipeline.get_coreml_pipe(
        pytorch_pipe=pytorch_pipe,
        mlpackages_dir=args.mldir,
        model_version=args.model,
        compute_unit=args.compute_unit,
        delete_original_pipe=True,
    )
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # 3. Generate
    print("Generating image...")
    t1 = time.time()
    np.random.seed(args.seed)
    result = coreml_pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
    )
    elapsed = time.time() - t1

    # 4. Save
    image = result.images[0]
    image.save(args.output)
    print(f"\nDone in {elapsed:.1f}s — saved {image.size[0]}x{image.size[1]} image to {args.output}")


if __name__ == "__main__":
    main()
