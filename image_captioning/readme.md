# Image Captioning with Visual Attention

The code is based on <a href="https://www.tensorflow.org/tutorials/text/image_captioning">"Image captioning 
with visual attention"</a> tutorial.


## Python Environment
In order to create the necessary python environment, execute the following lines:
```shell
conda create -n image_captioning python=3.8
conda activate image_captioning
pip install -r requirements.txt
```
If you want to work with a GPU, then you have to additionally install cudnn and cudatoolkit.
```shell
conda install cudnn cudatoolkit
```


## How to run the code
The following steps need to be taken to train and inspect a visual attention model.
1. Run notebook `image_captioning/notebooks/01-data-analysis.ipynb` to download the data.
2. Execute `image_captioning/scripts/encode_images_and_captions.py` to prepare image encodings and a text vectorizer for training.
3. Execute `image_captioning/scripts/train.py` to train the model.
4. Work with notebook `image_captioning/notebooks/02-model-inference.ipynb` to run inference and to visualize the attentions.

Note that the first three steps can consume some time to execute.
