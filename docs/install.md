# Installation

To install ``subdomain`` from [PyPI](https://pypi.org/project/subdomain/) just run

```
pip install subdomain
```

or alternatively install from [conda-forge](https://anaconda.org/conda-forge/subdomain)

```
conda install conda-forge::subdomain
```

## GPU support

For most users GPU support can be installed via

```
pip install 'subdomain[cuda12]' --extra-index-url=https://pypi.nvidia.com
```

However, this may not work in all circumstances (depending on the system setup and
available CUDA installations). Therefore, we recommend to follow the respective install
instructions

- [JAX](https://docs.jax.dev/page/installation.html) for accelerated convolutions (including TPU)
- [RAPIDS](https://docs.rapids.ai/install/) (only `cuML` is required) for accelerated KMeans
