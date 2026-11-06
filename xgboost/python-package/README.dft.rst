=========================
xgb-mps Python bindings
=========================

Porting XGBoost to MPS, by Codex.

This is the Python package from the independent `xgb-mps fork
<https://github.com/mmaaz-git/xgb-mps>`_ of XGBoost 3.4.2. It adds experimental
Apple GPU support through custom Metal kernels selected with ``device="mps"``.
The fork ships Python bindings only; R, Java/Scala and Spark bindings, including
``xgboost.spark``, are removed. The shared C++ engine and CPU/CUDA backends remain.

Installation
============

Build and install the fork using its `installation instructions
<https://github.com/mmaaz-git/xgb-mps/blob/main/README.md>`_. Metal requires a
Metal-enabled library build on macOS with an Apple GPU. The standard upstream
``pip install xgboost`` command does not install this fork.

After installing the Metal-enabled build, the Python API uses:

.. code-block:: python

    import xgboost as xgb

    model = xgb.XGBRegressor(device="mps", tree_method="hist")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

Status and attribution
======================

Testing has concentrated on an Apple M3. Broader hardware validation, NVIDIA
comparisons and performance optimization remain; a general CPU speedup is not
established. See the fork's `support summary
<https://github.com/mmaaz-git/xgb-mps/blob/main/docs/CUDA_PARITY.md>`_ and
`validation records
<https://github.com/mmaaz-git/xgb-mps/blob/main/docs/VALIDATION.md>`_.

Based on `XGBoost <https://github.com/dmlc/xgboost>`_, developed by its upstream
contributors. Licensed under Apache-2.0; the package includes ``LICENSE``.
