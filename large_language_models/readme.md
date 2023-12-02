# Large Language Models


Large language models (LLMs) like OpenAI's GPT series represent a significant breakthrough in artificial 
intelligence, primarily focusing on understanding and generating human language. These models are trained 
on extensive datasets and use advanced neural networks to perform a wide range of language-related tasks, 
from text completion to complex problem-solving. While they offer remarkable benefits in areas like content 
creation, automation, and translation, LLMs also pose challenges, including data bias and ethical concerns, 
necessitating careful consideration in their development and application.
In fact, this introduction was written by an LLM


## Python Environment
In order to run the code, simply create the python environment. You can use `conda` or `mamba` to create the environment.
```shell
conda env create -f environment.yaml
```


## Disclaimer
Note that there's a reason that LLMs are called large. The size of these models can be hundreds of GBs which
of course makes it impossible to run on my simple 32GB CPU RAM laptop. Smaller LLMs do exist, we'll try to use 
some of those. In addition, running inference with these models can be extremely slow especially on CPU. 
