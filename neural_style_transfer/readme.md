# Neural Style Transfer


The code is based on <a href="https://www.tensorflow.org/tutorials/generative/style_transfer">this</a> 
tensorflow tutorial.

## Make your own creation
To start the script in the background for generating an image, you
can use (make sure you have activated the right python environment):
```bash
nohup python YOUR_COMMAND > YOUR_LOG_FILE.log 2>&1 &
```
for example:
```bash
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
```
