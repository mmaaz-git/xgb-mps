# XGBoost engine for xgb-mps

This directory contains the vendored XGBoost 3.4.2 engine and the experimental
Metal backend used by the [xgb-mps fork](../README.md).

**This fork ships Python bindings only.** The shared C++ engine and C API remain;
R, Java/Scala and Spark bindings, including `xgboost.spark`, have been removed.
CPU and CUDA implementations remain alongside Metal.

Use the [project README](../README.md) for installation and `device="mps"`
examples, and the [support summary](../docs/CUDA_PARITY.md) for tested behavior
and limits. Upstream XGBoost packages and documentation describe the original
project, not this fork's Metal support.

## Source guide

- `src/` and `include/`: shared engine, backends and C API.
- `python-package/`: Python bindings and packaging.
- `tests/`: retained upstream C++ and Python tests; fork-specific Metal tests
  live in the repository's top-level [`tests/`](../tests/).
- `doc/` and `demo/`: retained upstream documentation and examples, with removed
  binding sections pruned.
- `dmlc-core/` and `plugin/`: dependencies and optional engine components.

## Upstream credits and license

XGBoost is developed by the [upstream contributors](CONTRIBUTORS.md).
See the [upstream repository](https://github.com/dmlc/xgboost) and
[documentation](https://xgboost.readthedocs.io/).

Reference: Tianqi Chen and Carlos Guestrin,
[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754),
KDD 2016. XGBoost originated at the University of Washington.

© Contributors, 2021. Licensed under [Apache-2.0](LICENSE).
Third-party notices and licenses remain with their source files.
