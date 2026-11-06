##################
Installation Guide
##################

XGBoost provides binary packages for some language bindings.  The binary packages support
the GPU algorithm (``device=cuda:0``) on machines with NVIDIA GPUs. Please note that
**training with multiple GPUs is only supported for Linux platform**. See
:doc:`gpu/index`.  Also we have both stable releases and nightly builds, see below for how
to install them.  For building from source, visit :doc:`this page </build>`.

.. contents:: Contents

Stable Release
==============

Python
------

Pre-built binary wheels are uploaded to PyPI (Python Package Index) for each release. Supported platforms are Linux (x86_64, aarch64), Windows (x86_64, aarch64) and MacOS (x86_64, Apple Silicon).

.. code-block:: bash

  pip install xgboost


You might need to run the command with ``--user`` flag or use ``virtualenv`` if you run into permission errors.

.. note:: Windows users need to install Visual C++ Redistributable

  XGBoost requires DLLs from `Visual C++ Redistributable
  <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_
  in order to function, so make sure to install it. Exception: If
  you have Visual Studio installed, you already have access to
  necessary libraries and thus don't need to install Visual C++
  Redistributable.

Capabilities of binary wheels for each platform:

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+---------------------+---------+----------------------+
| Platform            | GPU     | Multi-Node-Multi-GPU |
+=====================+=========+======================+
| Linux x86_64        | |tick|  |  |tick|              |
+---------------------+---------+----------------------+
| Linux aarch64       | |tick|  |  |tick|              |
+---------------------+---------+----------------------+
| MacOS x86_64        | |cross| |  |cross|             |
+---------------------+---------+----------------------+
| MacOS Apple Silicon | |cross| |  |cross|             |
+---------------------+---------+----------------------+
| Windows             | |tick|  |  |cross|             |
+---------------------+---------+----------------------+
| Windows aarch64     | |cross| |  |cross|             |
+---------------------+---------+----------------------+

Linux aarch64 wheels now ship with CUDA support, so ``pip install xgboost`` on modern
Jetson or Graviton machines provides the same GPU functionality as the Linux x86_64
wheel.

.. _wheel-cuda:

CUDA toolkit variants (Linux)
*****************************
The default ``xgboost`` wheel for Linux x86_64 and aarch64 is built with CUDA Toolkit 13.x:

.. code-block:: bash

  pip install xgboost

Users with GPU whose NVIDIA driver supports CUDA 12 but not CUDA 13 can instead install the CUDA 12 package:

.. code-block:: bash

  pip uninstall xgboost xgboost-cu12
  pip install xgboost-cu12

The CUDA 12 package is a driver-compatibility option.


Minimal installation (CPU-only)
*******************************
The default installation with ``pip`` will install the full XGBoost package, including support for GPU algorithms.

You may choose to reduce the size of the installed package and save the disk space, by opting to install ``xgboost-cpu`` instead:

.. code-block:: bash

  pip install xgboost-cpu

The ``xgboost-cpu`` variant has a drastically smaller disk footprint, but does not provide GPU algorithms.

Conda
*****

You may use the Conda packaging manager to install XGBoost:

.. code-block:: bash

   conda install -c conda-forge py-xgboost

Conda should be able to detect the existence of a GPU on your machine and install the correct variant of XGBoost. If you run into issues, try indicating the variant explicitly:

.. code-block:: bash

   # CPU variant
   conda install -c conda-forge py-xgboost=*=cpu*
   # GPU variant
   conda install -c conda-forge py-xgboost=*=cuda*

To force the installation of the GPU variant on a machine that does not have an NVIDIA GPU, use environment variable ``CONDA_OVERRIDE_CUDA``,
as described in `"Managing Virtual Packages" in the conda docs <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html>`_.

.. code-block:: bash

  export CONDA_OVERRIDE_CUDA="12.8"
  conda install -c conda-forge py-xgboost=*=cuda*

You can install Conda from the following link: `Download the conda-forge Installer <https://conda-forge.org/download/>`_.

Nightly Build
=============


Python
------

Nightly builds are available. You can go to `this page <https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html>`_,
find the wheel with the commit ID you want and install it with pip:

.. code-block:: bash

  pip install <url to the wheel>


The capability of Python pre-built wheel is the same as stable release.
