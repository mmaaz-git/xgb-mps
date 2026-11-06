===================================
xgb-mps Python bindings: CPU only
===================================

This is the CPU-only Python package variant retained by the `xgb-mps fork
<https://github.com/mmaaz-git/xgb-mps>`_ of XGBoost. It excludes GPU algorithms
and federated learning. It does not provide ``device="mps"`` support.

For a Metal-enabled Python build, follow the fork's `installation instructions
<https://github.com/mmaaz-git/xgb-mps/blob/main/README.md>`_ instead. The upstream
``pip install xgboost-cpu`` command installs the upstream distribution, not this
fork.

This fork ships Python bindings only. R, Java/Scala and Spark bindings,
including ``xgboost.spark``, are removed. The CPU package variant retains the
shared C++ engine and the usual Python API.

Based on `XGBoost <https://github.com/dmlc/xgboost>`_, developed by its upstream
contributors. Licensed under Apache-2.0; the package includes ``LICENSE``.
