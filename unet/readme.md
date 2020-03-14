# U-net for Semantic Segmentation


# Introduction
The data used to try out this model, can be found on 
<a href="https://www.kaggle.com/c/tgs-salt-identification-challenge/data">kaggle</a>.
The code and logic is inspired by 
<a href="https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47">this</a> 
blog post. Extended U-net (with residual blocks) can be found in 
<a href="https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss">this</a> 
kaggle notebook.


# Data
In order to directly be able to run all the code, we have to put the data in the correct
location; create a `data/train` directory under the `computer_vision/unet` folder and 
unzip the `train.zip` here. After unzipping, you should have two folders:
* `data/train/images/`
* `data/train/masks/`


# Training
To start a training, select and optionally configure a configurations
file from the `unet/model/configs/` directory and execute the 
`unet/model/train.py` script with `unet/model/` as the working 
directory.
