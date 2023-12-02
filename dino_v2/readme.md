# DINO


<a href="https://github.com/facebookresearch/dinov2">DINOv2 github</a>


## Python Environment
In order to run the code, simply create the python environment. The environment has been copied from the DINOv2
github repo. You can use `conda` or `mamba` to create the environment.
```shell
conda env create -f environment.yaml
```
This will create a python environment called 'dinov2'. You can additionally install `ipykernel` to run notebooks 
with this env.
```bash
conda activate dinov2
conda install ipykernel
```


## DINOv1 and DINOv2
There's a notebook for both DINOv1 and for DINOv2. DINOv1 was added since it allows for easy extraction 
of the attention values and consequently for visualization.
