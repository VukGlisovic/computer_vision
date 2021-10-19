#!/usr/bin/env bash


# activate the correct python environment
source activate computer_vision

# execute the script
nohup python generate_stylized_image.py \
    -c=CONTENT_IMAGE_PATH \
    -s=STYLE_IMAGE_PATH \
    -o=OUTPUT_IMAGE_PATH \
    -md=1024 \
    -a=1 \
    -b=1000 \
    -vw=3 \
    -lr=0.01 \
    -ep=1000 \
    -st=50 \
    -wn \
> YOUR_LOG_FILE.log 2>&1 &
