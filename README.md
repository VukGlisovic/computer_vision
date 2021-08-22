# Computer Vision

## Installation Requirements

This repository is for experimenting with and learning about different 
computer vision methods/models. In particular we'll use tensorflow 2.x
to learn about that as well. The `environment.yaml` contains all the 
packages to run any of the scripts/notebooks in this repo. Use this 
environment unless stated otherwise.

In order to run code, you have to add the repository to your `PYTHONPATH` 
(in your `.bashrc`):
```bash
export PYTHONPATH="${PYTHONPATH}:<PATH_TO_REPOSITORY>/computer_vision"
```


## Classification Models

#### LeNet-5
This was the first in a series of convolutional neural network architectures.
In their <a href="http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf">paper</a> 
from 1998, they describe a, nowadays considered small, neural network using 
only valid convolutions.

#### AlexNet
The next famous neural network for image classification, was AlexNet. Here
is the <a href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">paper</a>
from 2012. This network started using ReLU activations and same convolutions.

#### VGG - 16
In this <a href="https://arxiv.org/pdf/1409.1556.pdf">paper</a> from 2015,
showed systematic decreases in height and width and increase in the number
of channels which made it appealing.

#### ResNet-50
ResNet started using skip-connections. It is making use of residual blocks.
Here is the <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf">paper</a>
from 2015. A residual block basically applies this formula:

a<sup>l+2</sup> = g(z<sup>l+2</sup> + a<sup>l</sup>)

where l is the layer number. Basically, it could learn an identity mapping
from layer l to layer l+2 by mitigating z<sup>l+2</sup> and by using ReLU
activations. This allowed networks to grow much much deeper.

#### Inception Network (GooLeNet)
Here the authors decided to use multiple filter sizes (1x1, 3x3, 5x5). Also
a same max pool was used. The outputs of all these convolutions and the pooling
are concatenated across the number of channels. Additionally, it applies
a 1x1 convolution before a 3x3 and 5x5 convolution to shrink the number of
channels and therefore reduce the number of trainable weights. This 1x1 
convolution is called the bottleneck layer. Here's the 
<a href="https://static.googleusercontent.com/media/research.google.com/nl//pubs/archive/43022.pdf">paper</a>
from 2014.

#### EfficientNet
This family of architectures (EfficientNetB0, ..., EfficientNetB7), is a new approach
to model scaling. It is based on https://arxiv.org/pdf/1905.11946.pdf


## Detection Models

#### YOLO
You Only Look Once (YOLO) is a object detection network from this
<a href="https://arxiv.org/pdf/1506.02640.pdf">paper</a>. In this
network, the approach is to divide the input image in a (for example 
19x19) grid, and then for each grid try to detect an object. The reason 
why this network is popular, is because of its performance; inference
time is very low.

#### EfficientDet
The implementation in `computer_vision/efficientdet`, is based on 
https://arxiv.org/pdf/1911.09070.pdf and https://keras.io/examples/vision/retinanet/#computing-pairwise-intersection-over-union-iou.


## Segmentation Models

#### Unet
This architecture is called Unet because it is shape like a U. This is
nicely depicted here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/


## Neural Style Transfer
This part of computer vision is an extremely interesting and fun
application that combines two images (a content and a style image)
and produces one image that has the content of the content image,
but drawn in the style of the style image. This will make you 
create pictures like picasso. The corresponding <a href="https://arxiv.org/pdf/1508.06576.pdf">paper</a> 
is from 2015.
