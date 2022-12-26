# BlazeFace
### Sub-millisecond Neural Face Detection on Mobile GPUs

BlazeFace is a model that detects both the face bounding box together with a few
landmarks on the face. The <a href="https://arxiv.org/pdf/1907.05047.pdf">"paper"</a> 
is a nice short, but nevertheless interesting one to read.


## Python Environment
In order to create the necessary python environment, execute the following lines:
```shell
conda create -n blazeface python=3.8
conda activate blazeface
pip install -r requirements.txt
```
If you want to work with a GPU, then you have to additionally install cudnn and cudatoolkit.
```shell
conda install cudnn cudatoolkit
```

