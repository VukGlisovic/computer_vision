#!/usr/bin/env bash

# make sure the dreambooth python env is activated
autotrain dreambooth \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --project-name "vuk-SDXL" \
  --image-path "../data/vuk-photos/crops" \
  --prompt "A photo of Vuk his smiling face and wearing a cap on his head." \
  --resolution 1024 \
  --batch-size 1 \
  --num-steps 500 \
  --gradient-accumulation 4 \
  --lr 1e-4 \
  --mixed-precision "fp16"
