.. image:: _static/logo.png
  :align: center
  :alt: BOOMER: Gradient Boosted Multi-Label Classification Rules


Overview
========

BOOMER is an algorithm for learning ensembles of gradient boosted multi-label classification rules that integrates with the popular `scikit-learn <https://scikit-learn.org>`_ machine learning framework. It allows to train a machine learning model on labeled training data, which can afterwards be used to make predictions for unseen data. In contrast to prominent boosting algorithms like `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_ or `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_, the algorithm is aimed at `multi-label classification <https://en.wikipedia.org/wiki/Multi-label_classification>`_ problems, where individual data examples do not only correspond to a single class, but may be associated with several labels at the same time. Real-world applications of multi-label classification include the assignment of keywords to text documents, the annotation of multimedia data, such as images, videos or audio recordings, as well as applications in the field of biology, chemistry and more. To provide a versatile tool for different use cases, great emphasis is put on the efficiency of the implementation. To ensure its flexibility, it is designed in a modular fashion and can therefore easily be adjusted to different requirements.

This document is intended for users and developers who are interested in the algorithm's implementation. For a detailed description of the used methodology, please refer to the publications that are listed in the :ref:`references` section of this documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart/index

   testbed/index

   api/index

   references/index

   source/CHANGELOG.md

   source/CONTRIBUTORS.md

   source/LICENSE.md
