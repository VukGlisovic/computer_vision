#!/usr/bin/env bash

export USE_8BIT_ADAM="False"
export USE_XFORMERS="False"
export TRAIN_TEXT_ENCODER="False"
export DISABLE_GRADIENT_CHECKPOINTING="False"

# make sure the dreambooth python env is activated
autotrain dreambooth \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --project-name "vuk-SDXL" \
  --image-path "../data/vuk-photos/crops/" \
  --prompt "A photo of Vuk Glisovic wearing casual clothes, taking a selfie, and smiling." \
  --resolution 1024 \
  --batch-size 1 \
  --num-steps 500 \
  --gradient-accumulation 4 \
  --lr 1e-4 \
  --mixed-precision "fp16"
