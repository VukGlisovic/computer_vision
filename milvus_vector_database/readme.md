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

### Create a docker container for Milvus
In order to use the remote version of Milvus (instead of the Lite) version, use the 
`milvus_vector_database/docker/standalone_embed.sh` script to create a docker container with the Milvus vector database
running inside the container. More info can be found on [milvus with docker](https://milvus.io/docs/install_standalone-docker.md).
To access the WebUI, access http://127.0.0.1:9091/webui/.

Note that the lite version of Milvus allows for only a small subset of vector indexes. So if you want to use more
advanced indexing procedures like Google's scann, you'll have to use the remote variant.
