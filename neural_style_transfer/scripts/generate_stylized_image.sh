#!/usr/bin/env bash


# activate the correct python environment
source activate computer_vision

# execute the script
nohup python generate_stylized_image.py \
    -c=CONTENT_IMAGE_PATH \
    -s=STYLE_IMAGE_PATH \
    -o=OUTPUT_IMAGE_PATH \
    -a=1000 \
    -b=1 \
    -vw=3 \
    -lr=0.02 \
    -wn \
> YOUR_LOG_FILE.log 2>&1 &
