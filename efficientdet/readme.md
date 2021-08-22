# EfficientDet


The code is based on a <a href="https://keras.io/examples/vision/retinanet/#computing-pairwise-intersection-over-union-iou">retinanet</a> 
tensorflow tutorial. The most important part is that the input pipeline is correct
where bounding boxes, labels are correctly encoded with their images. Afterwards
the decoding is of course very important during training, but also during inference.

## Train a model
The following two steps need to be executed to train a model.
1. create data; run `efficientdet/notebooks/01-create-mnist-bbox-data.ipynb`
2. train model; run `efficientdet/scripts/train.py`

This will create artifical data; it basically draws mnist digits into an
image and stores the corresponding bounding box. After training the model,
you'll have a simple mnist digits object detection model that detects all
digits from 0 to 9.
