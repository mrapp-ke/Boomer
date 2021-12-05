.. _usage:

Using the Algorithm
-------------------

The BOOMER algorithm is implemented by the class ``Boomer`` that is part of the `mlrl-boomer <https://pypi.org/project/mlrl-boomer/>`__ package (see :ref:`installation`). As it follows the conventions of a scikit-learn `estimator <https://scikit-learn.org/stable/glossary.html#term-estimators>`_, it can be used similarly to other classification methods that are included in this popular machine learning framework. The `getting started guide <https://scikit-learn.org/stable/getting_started.html>`_ that is provided by the scikit-learn developers is a good starting point for learning about the framework's functionalities and how to use them.

**Fitting an estimator to training data**

An illustration of how the algorithm can be fit to exemplary training data is shown in the following:

.. code-block:: python

   from mlrl.boosting import Boomer

   clf = Boomer()  # Create a new estimator
   x = [[  1,  2,  3],  # Two training examples with three features
        [ 11, 12, 13]]
   y = [[1, 0],  # Ground truth labels of each training example
        [0, 1]]
   clf.fit(x, y)

The ``fit`` method accepts two inputs, ``x`` and ``y``:

* A two-dimensional feature matrix ``x``, where each row corresponds to a training example and each column corresponds to a particular feature.
* An one- or two-dimensional binary feature matrix ``y``, where each row corresponds to a training example and each column corresponds to a label. If an element in the matrix is unlike zero, it indicates that the respective label is relevant to an example. Elements that are equal to zero denote irrelevant labels. In multi-label classification, where each example may be associated with several labels, the label matrix is two-dimensional. However, the BOOMER algorithm is also capable of dealing with traditional binary classification problems, where an one-dimensional vector of ground truth labels is provided to the learning algorithm.

Both, ``x`` and ``y``, are expected to be `numpy arrays <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ or equivalent `array-like <https://scikit-learn.org/stable/glossary.html#term-array-like>`_ data types. In addition, the BOOMER algorithm does also support to use `scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

In the previous example the BOOMER algorithm's default configuration is used. However, in many cases it is desirable to adjust its behavior by providing custom values for one or several of its parameters. This can be achieved by passing the names and values of the respective parameters as constructor arguments:

.. code-block:: python

   clf = Boomer(max_rules=100, loss='logistic_example_wise')

A description of all available parameters is available in the section :ref:`parameters`.

**Using an estimator for making predictions**

Once the estimator has been fitted to the training data, its ``predict`` method can be used to obtain predictions for previously unseen examples:

.. code-block:: python

   pred = clf.predict(x)
   print(pred)

In this example, we use the estimator to predict for the same data that has previously been used for training. This results in the original ground truth labels to be printed:

.. code-block:: python

   [[1 0]
    [0 1]]

In practice, one usually retrieves the data from files rather than manually specifying the values of the feature and label matrices. A collection of benchmark datasets can be found `here <https://github.com/mrapp-ke/Boomer-Datasets>`_.
