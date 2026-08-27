# Super Resolution based Compression


This project is based on the paper "Lossless Image Compression through Super-Resolution" (https://arxiv.org/pdf/2004.02872). The code is based on https://github.com/caoscott/SReC.

The idea behind this paper, is to train a model that can basically predict the next pixel based on neighbouring pixels. By using such a trained model, we can compress images significantly while preserving the exact pixel values. It is basically the same as storing a file in PNG format (which is also a lossless compression algorithm), except that it should be even more effecient and thus the file size should be smaller. The downside of course, is that you need the model to decode the compressed image again.
