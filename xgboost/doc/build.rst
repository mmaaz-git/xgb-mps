####################
Building From Source
####################

This page gives instructions on how to build and install XGBoost from the source code on
various systems.  If the instructions do not work for you, please feel free to ask
questions at `GitHub <https://github.com/dmlc/xgboost/issues>`__.

.. note:: Pre-built binary is available: now with GPU support

  Consider installing XGBoost from a pre-built binary, to avoid the trouble of building XGBoost from the source.  Checkout :doc:`Installation Guide </install>`.

.. contents:: Contents
  :local:

.. _get_source:

*************************
Obtaining the Source Code
*************************

To obtain the development repository of XGBoost, one needs to use ``git``. XGBoost uses
Git submodules to manage dependencies. So when you clone the repo, remember to specify
``--recursive`` option:

  .. code-block:: bash

    git clone --recursive https://github.com/dmlc/xgboost

.. _build_shared_lib:

***************************
Building the Shared Library
***************************

This section describes the procedure to build the shared library and CLI interface
independently. For building language specific package, see corresponding sections in this
document.

- On Linux and other UNIX-like systems, the target library is ``libxgboost.so``
- On MacOS, the target library is ``libxgboost.dylib``
- On Windows the target library is ``xgboost.dll``

This shared library is used by different language bindings (with some additions depending
on the binding you choose).  The minimal building requirement is

- A recent C++ compiler supporting C++17. We use gcc, clang, and MSVC for daily
  testing.
- CMake 3.18 or higher.

For a list of CMake options like GPU support, see ``#-- Options`` in CMakeLists.txt on top
level of source tree. We use ``ninja`` for build in this document, specified via the CMake
flag ``-GNinja``. If you prefer other build tools like ``make`` or ``Visual Studio 17
2022``, please change the corresponding CMake flags. Consult the `CMake generator
<https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html>`_ document when
needed.

.. _running_cmake_and_build:

Running CMake and build
=======================

After obtaining the source code, one builds XGBoost by running CMake:

.. code-block:: bash

  cd xgboost
  cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
  cd build && ninja


The same command applies for both Unix-like systems and Windows. After running the
build, one should see a shared object under the ``xgboost/lib`` directory.

- Building on MacOS

  On MacOS, one needs to obtain ``libomp`` from `Homebrew <https://brew.sh/>`_ first:

  .. code-block:: bash

    brew install libomp

- Visual Studio

  The latest Visual Studio has builtin support for CMake projects. If you prefer using an
  IDE over the command line, you can use the ``open with visual studio`` option in the
  right-click menu under the ``xgboost`` source directory. Consult the VS `document
  <https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170>`__
  for more info.

.. _build_gpu_support:


Building with GPU support
=========================

XGBoost can be built with GPU support for both Linux and Windows using CMake. See
`Building R package with GPU support`_ for special instructions for R.

An up-to-date version of the CUDA toolkit is required.

.. note:: Checking your compiler version

    CUDA is really picky about supported compilers, a table for the compatible compilers
    for the latest CUDA version on Linux can be seen `here
    <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_.

Some distros package a compatible ``gcc`` version with CUDA. If you run into compiler
errors with ``nvcc``, try specifying the correct compiler with
``-DCMAKE_CXX_COMPILER=/path/to/correct/g++ -DCMAKE_C_COMPILER=/path/to/correct/gcc``. On
Arch Linux, for example, both binaries can be found under ``/opt/cuda/bin/``. In addition,
the ``CMAKE_CUDA_HOST_COMPILER`` parameter can be useful.

From the command line on Linux starting from the XGBoost directory, add the ``USE_CUDA``
flag:

.. code-block:: bash

  cmake -B build -S . -DUSE_CUDA=ON -GNinja
  cd build && ninja

To speed up compilation, the compute version specific to your GPU could be passed to cmake
as, e.g., ``-DCMAKE_CUDA_ARCHITECTURES=75``. A quick explanation and numbers for some
architectures can be found `in this page <https://developer.nvidia.com/cuda-gpus>`_.

- Faster distributed GPU training with NCCL

  By default, distributed GPU training is enabled with the option
  ``USE_NCCL=ON``. Distributed GPU training depends on NCCL2, available at `this link
  <https://developer.nvidia.com/nccl>`_. Since NCCL2 is only available for Linux machines,
  **Distributed GPU training is available only for Linux**.

  .. code-block:: bash

    cmake -B build -S . -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT=/path/to/nccl2 -GNinja
    cd build && ninja

  Some additional flags are available for NCCL, ``BUILD_WITH_SHARED_NCCL`` enables
  building XGBoost with NCCL as a shared library, while ``USE_DLOPEN_NCCL`` enables
  XGBoost to load NCCL at runtime using ``dlopen``.

.. _build_python:

***********************************
Building Python Package from Source
***********************************

The Python package is located at ``python-package/``.

Building Python Package with Default Toolchains
===============================================
There are several ways to build and install the package from source:

1. Build C++ core with CMake first

  You can first build C++ library using CMake as described in :ref:`build_shared_lib`.
  After compilation, a shared library will appear in ``lib/`` directory.
  On Linux distributions, the shared library is ``lib/libxgboost.so``.
  The install script ``pip install .`` will reuse the shared library instead of compiling
  it from scratch, making it quite fast to run.

  .. code-block:: console

    $ cd python-package/
    $ pip install .  # Will re-use lib/libxgboost.so

2. Install the Python package directly

  If the shared object is not present, the Python project setup script will try to run the
  CMake build command automatically. Navigate to the ``python-package/`` directory and
  install the Python package by running:

  .. code-block:: console

    $ cd python-package/
    $ pip install -v .

  which will invoke CMake underneath to build the shared library
  (``libxgboost.so`` on Linux). To customize the CMake options,
  add the command-line argument
  ``--config-settings cmake.define.[CMAKE_OPTION]=[OPTION_VALUE]``.
  (Pip 22.1+ is required.)

  For example:

  .. code-block:: console

    $ pip install -v . \
        --config-settings cmake.define.USE_CUDA=ON \
        --config-settings cmake.define.USE_NCCL=ON

  Common CMake options (see :ref:`build_shared_lib` for the full list):

  - ``USE_CUDA`` — build with CUDA / GPU acceleration
  - ``USE_NCCL`` — build with NCCL for distributed GPU training
  - ``USE_DLOPEN_NCCL`` — load NCCL dynamically at runtime
  - ``PLUGIN_FEDERATED`` — enable the federated learning plugin
  - ``PLUGIN_RMM`` — build with RAPIDS Memory Manager support
  - ``HIDE_CXX_SYMBOLS`` — hide all C++ symbols in the shared library
  - ``USE_OPENMP`` — build with OpenMP (defaults to ON)
  - ``XGBOOST_USE_SYSTEM_LIBXGBOOST`` — see Item 4 below

  .. note:: Verbose flag recommended

    As ``pip install .`` will build C++ code, it will take a while to complete.
    To ensure that the build is progressing successfully, we suggest that
    you add the verbose flag (``-v``) when invoking ``pip install``.


3. Editable installation

  To further enable rapid development and iteration, we provide an **editable
  installation**.  In an editable installation, the installed package is simply a symbolic
  link to your working copy of the XGBoost source code. So every changes you make to your
  source directory will be immediately visible to the Python interpreter. To install
  XGBoost as editable installation, first build the shared library as previously described
  in :ref:`running_cmake_and_build`, then install the Python package with the ``-e`` flag:

  .. code-block:: bash

    # Build shared library libxgboost.so
    cmake -B build -S . -GNinja
    cd build && ninja
    # Install as editable installation
    cd ../python-package
    pip install -e .

4. Reuse the ``libxgboost.so`` on system path.

  This option is useful for package managers that wish to separately package
  ``libxgboost.so`` and the XGBoost Python package. For example, Conda
  publishes ``libxgboost`` (for the shared library) and ``py-xgboost``
  (for the Python package).

  To use this option, first make sure that ``libxgboost.so`` exists in the system library path:

  .. code-block:: python

    import sys
    import pathlib
    libpath = pathlib.Path(sys.base_prefix).joinpath("lib", "libxgboost.so")
    assert libpath.exists()

  Then pass ``cmake.define.XGBOOST_USE_SYSTEM_LIBXGBOOST=ON`` to ``pip install``:

  .. code-block:: bash

    cd python-package
    pip install . --config-settings cmake.define.XGBOOST_USE_SYSTEM_LIBXGBOOST=ON


.. note::

  See :doc:`contrib/python_packaging` for instructions on packaging and distributing
  XGBoost as Python distributions.


**************************
Building the Documentation
**************************

XGBoost uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ for documentation.  To
build it locally, you need a installed XGBoost with all its dependencies along with:

* System dependencies

  - git
  - graphviz

* Python dependencies

  Checkout the ``requirements.txt`` file under ``doc/``

Under ``xgboost/doc`` directory, run ``make <format>`` with ``<format>`` replaced by the
format you want.  For a list of supported formats, run ``make help`` under the same
directory. This builds a partial document for Python but not other language bindings. To
build the full document, see :doc:`/contrib/docs`.
