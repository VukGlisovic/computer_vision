# YOLO


This <a href="https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/">blog</a> 
was the main guideline in creating this code.

The open source project we'll use is 
<a href="https://github.com/experiencor/keras-yolo3">experiencor</a>. 
In the rest of the readme, I will refer to this repository as experiencor. 

#### 1. Download git repository
Clone the repository into a directory called `vendor`.
```bash
# navigate to computer_vision/yolo
mkdir vendor
cd vendor
git clone https://github.com/experiencor/keras-yolo3.git
```

#### 2. Create python environment
In experiencor, you can find a `requirements.txt`. Use this to create
the python environment. I use conda to manage my python environments, but
you can use anything you like. Note that creating and installing the required 
packages may take some time. 
```bash
conda create -n yolo python=3.6
conda activate yolo
pip install -r requirements.txt
sudo apt-get install graphviz  # for keras model visualizations
pip install pydot==1.4.1 graphviz==0.14  # for keras model visualizations
conda install ipykernel  # to be able to use the environment in jupyter
```
Use this environment to run any code in the `yolo` folder.


#### 3. Download weights
We can download pretrained weights from https://pjreddie.com/media/files/yolov3.weights.
Put these weights in folder `yolo/models/MSCOCO/`.


#### 4. Create keras model
Next, we need to define a keras model that has the correct architecture: 
thus the right number and type of layers to match the downloaded 
model weights.

To create the keras model, run `yolo/scripts/create_keras_model.py`

You might get an error from `np.set_printoptions(threshold=np.nan)`; you can
replace `np.nan` by a large number (e.g. `sys.maxsize`) to fix this.
