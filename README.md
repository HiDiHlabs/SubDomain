# SubDomain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Docs](https://img.shields.io/badge/Docs-online-blue)](https://HiDiHlabs.github.io/SubDomain/latest/)
<!--
[![PyPI](https://img.shields.io/pypi/v/subdomain)](https://pypi.org/project/subdomain)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/subdomain/README.html)
-->


``subdomain`` allows identification of domains from segmentation-free analysis of
spatially-resolved transcriptomics. More generally it can be applied to any labeled 2D
grid data.

## Installation

`subdomain` can currently be installed from GitHub via

```sh
pip install git+https://github.com/HiDiHlabs/SubDomain.git
```

<!--
`subdomain` is available on [PyPI](https://pypi.org/project/subdomain/) and
[bioconda](https://bioconda.github.io/recipes/subdomain/README.html).

```sh
# PyPI
pip install subdomain
```

```sh
# or conda
conda install bioconda::subdomain
```
-->

For detailed installation instructions please refer to the
[documentation](https://HiDiHlabs.github.io/SubDomain/latest/install.html).

## Documentation

For documentation of the package please refer to
[GitHub pages](https://HiDiHlabs.github.io/SubDomain/latest/).

## Citations

tbd

# ToDo

- Reduce dependencies by using jax-based GMM library (e.g. https://github.com/adonath/gmmx ?); allows dropping pytorch dependencies
- Test benefits of GPU acceleration. Currently likely memory constrained. Switch to CPU-only clustering?

## License

This project is licensed under the MIT License - for details please refer to the
[LICENSE](./LICENSE) file.
