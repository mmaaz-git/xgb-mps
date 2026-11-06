##########################
Documentation and Examples
##########################

**Contents**

.. contents::
  :backlinks: none
  :local:

*************
Documentation
*************
* Python and C documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
* Each document is written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
* The documentation is the ``doc/`` directory.
* You can build it locally using ``make html`` command.

  .. code-block:: bash

    make html

  Run ``make help`` to learn about the other commands.

The online document is hosted by `Read the Docs <https://readthedocs.org/>`__ where the imported project is managed by `Hyunsu Cho <https://github.com/hcho3>`__ and `Jiaming Yuan <https://github.com/trivialfis>`__.

=========================================
Build the Python Docs using pip and Conda
=========================================

#. Create a conda environment.

   .. code-block:: bash

     conda create -n xgboost-docs --yes python=3.12

   .. note:: Python is required to match XGBoost's minimum supported Python version.

#. Activate the environment

   .. code-block:: bash

     conda activate xgboost-docs

#. Install required packages (in the current environment) using ``pip`` command.

   .. code-block:: bash

     pip install -r requirements.txt

   .. note::
      It is currently not possible to install the required packages using ``conda``
      due to ``xgboost_ray`` being unavailable in conda channels.

      .. code-block:: bash

        conda install --file requirements.txt --yes -c conda-forge


#. (optional) Install `graphviz <https://www.graphviz.org/>`__

   .. code-block:: bash

     conda install graphviz --yes

#. Eventually, build the docs.

   .. code-block:: bash

     make html

  You should see the following messages in the console:

  .. code-block:: console

    $ make html
    sphinx-build -b html -d _build/doctrees   . _build/html
    Running Sphinx v6.2.1
    ...
    The HTML pages are in _build/html.

    Build finished. The HTML pages are in _build/html.

*************
Read The Docs
*************

Setting ``READTHEDOCS=True`` enables the C API Doxygen build. Python API
documentation is generated locally by Sphinx; this fork does not fetch R or JVM
documentation artifacts.

********
Examples
********
* Use cases and examples are in `demo <https://github.com/dmlc/xgboost/tree/master/demo>`_ directory.
* We are super excited to hear about your story. If you have blog posts,
  tutorials, or code solutions using XGBoost, please tell us, and we will add
  a link in the example pages.

*********
Doc Tests
*********

We use Sphinx doctest to test selected snippets in the documentation. At the moment, this
only covers Python and R snippets written with ``.. code-tab:: python`` or ``.. code-tab::
r``. Regular code blocks are rendered as examples but are not executed by the doctest job.

The doctest job runs snippets from each ``.rst`` file as an independent group. Snippets in
the same document share state, while snippets from different documents do not. To skip a
tabbed snippet, add the ``no-doctest`` class:

.. code-block:: rst

  .. code-tab:: python
     :class: no-doctest

     # This snippet is rendered but not tested.
