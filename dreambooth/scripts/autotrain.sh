#!/usr/bin/env bash

conda activate dreambooth

autotrain dreambooth \
  --model $MODEL_NAME \
  --project-name $PROJECT_NAME \
  --image-path $DATA_DIR \
  --prompt "A photo of Vuk his smiling face and wearing a cap on his head." \
  --resolution 1024 \
  --batch-size 1 \
  --num-steps 500 \
  --gradient-accumulation 4 \
  --lr 1e-4 \
  --fp16 \
  --gradient-checkpointing
