.. _testbed:

Command Line API
================

As an alternative to using the BOOMER algorithm in your own Python program (see :ref:`usage`), the command line API that is provided by the package `mlrl-testbed <https://pypi.org/project/mlrl-testbed/>`__ (see :ref:`installation`) can be used to run experiments without the need to write code. Currently, it provides the following functionalities:

* The predictive performance in terms of commonly used evaluation measures can be assessed by using predefined splits of a dataset into training and test data or via `cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_.
* Experimental results can be written into output files. This includes evaluation scores, the predictions of a model, textual representations of rules, as well as the characteristics of models or datasets.
* Models can be stored on disk and reloaded for later use.

.. include:: experiments.inc.rst
.. include:: arguments.inc.rst
