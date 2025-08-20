# Milvus Vector Database

[Milvus](https://milvus.io/) is a high performance vector database. This project demonstrates various optimization 
techniques for vector search including quantization, advanced indexing algorithms like Google's ScaNN, and GPU acceleration.


## Quick Start

### Create Python Environment
```bash
# Run the following command from anywhere within the milvus_vector_database project
pixi install

# Or run scripts directly (auto-installs the environment)
pixi run load_glove
```

### Basic Usage
```python
from milvus_vector_database.src.milvus_glove import MilvusGlove, OptimizationPreset

# Create client and load data
client = MilvusGlove(remote=False)
client.create_collection(overwrite=True)
# ... insert vectors ...

# Apply optimization
client.optimize_with_preset("scann")  # or "speed", "memory", "balanced"

# Search with optimization
results = client.search_vectors(query_vector, k=10)
```

### Start a Milvus service with docker
The lite version of Milvus allows for only a small subset of vector indexes. So if you want to use more
advanced indexing procedures like Google's scann, you'll have to use the remote variant.

#### Docker with CPU
In order to use the remote version of Milvus (instead of the Lite) version, use the 
`milvus_vector_database/docker/cpu/standalone_embed.sh` script to create a docker container with the Milvus vector database
running inside the container. More info can be found on [milvus with CPU docker](https://milvus.io/docs/install_standalone-docker.md).
Start the container with 
```bash
cd milvus_vector_database/docker/cpu/
./standalone_embed.sh start
```

#### Docker with GPU
Some [index types need GPU](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/IndexType.md) to
work. The standard standalone version doesn't support GPU. For this you will need to use the GPU version of the docker
image. Detailed instructions can be found here: [milvus with GPU docker](https://milvus.io/docs/install_standalone-docker-compose-gpu.md).

Also note that standard Docker cannot use GPUs. You will also need to install a special toolkit from NVIDIA that acts as 
a bridge between Docker and your NVIDIA drivers. Follow the [NVIDIA Container Toolkit instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and restart the docker daemon in the end.

Start the docker containers with
```bash
cd milvus_vector_database/docker/gpu/
sudo docker compose up
```

#### WebUI
Milvus also has a web user interface. To access the WebUI go to http://127.0.0.1:9091/webui/.


## Benchmark Results

### Comparing different indexes
As part of the project, I wanted to run some benchmarks on various indexes. There's many things you can monitor when
benchmarking an index, like memory usage, CPU usage and more. But for this analysis, I only focused on speed and 
accuracy. In this first benchmark, I focused on different indexes with some default hyperparameters. To get more
info on which indexes the names in the plot refer to, I'd suggest to have a look at `milvus_vector_database/src/milvus/index_configuration.py`.

![Benchmark multiple indexes](resources/benchmark-results-multiple-indexes.jpg)

Some take-aways and notes from this plot:
* `accuracy` and `gpu_accuracy` both have 100% accuracy (or recall). This is expected as we're basically brute-forcing it.
* Note that `accuracy` takes more than 100ms per query (much more than the others), so I decided to crop it off of the plot.
* Nice to see that `gpu_accuracy` is almost 2 orders of magnitude faster than `accuracy` (I used an H100 GPU with 24GB of memory).
* Google's `scann` is the fastest of them all, however it does sacrifice quite some accuracy for low values of `k`.
* You can optimize for both memory usage and speed. The `balanced` approach tries to make sure we still use memory and are not the fastest,
  but as you can see it has some of the highest accuracies.

### Comparing different configurations of an index
I also wanted to just choose one index and play around with it. I decided to go with the `IVF_SQ8` index type because 
it's quite intuitive to configure and fast to evaluate.
