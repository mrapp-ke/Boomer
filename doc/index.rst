.. image:: _static/logo.png
  :align: center
  :alt: BOOMER: Gradient Boosted Multi-Label Classification Rules


Overview
========

BOOMER is an algorithm for learning gradient boosted multi-label classification rules. It allows to train a machine learning model on labeled training data, which can afterwards be used to make predictions for unseen data. In contrast to prominent boosting algorithms like `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_ or `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_, the algorithm is aimed at multi label classification problems, where individual data examples are not only associated with a single class, but may correspond to several labels at the same time.

This document is intended for users and developers that are interested in the algorithm's implementation. For a detailed description of the used methodology, please refer to the section :ref:`references`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart/index

   api/index

   references/index
