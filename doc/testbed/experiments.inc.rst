.. _experiments:

Running Experiments
-------------------

In the following, a minimal working example of how to use the command line API for applying the BOOMER algorithm to a particular dataset is shown:

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name

Both arguments that are included in the above command are mandatory:

* ``--data-dir``: The path of the directory where the data set files are located.
* ``--dataset``: The name of the data set files (without suffix).

The program expects the data set files to be provided in the `Mulan format <http://mulan.sourceforge.net/format.html>`_. It requires two files to be present in the specified directory:

#. An `.arff <http://weka.wikispaces.com/ARFF>`_ file that specifies the feature values and ground truth labels of the training examples.
#. An .xml file that specifies the names of the labels.

The Mulan dataset format is commonly used for benchmark datasets that allow to compare the performance of different machine learning approaches in empirical studies. A collection of publicly available benchmark datasets is available `here <https://github.com/mrapp-ke/Boomer-Datasets>`_.

If an .xml file is not provided, the program tries to retrieve the number of labels from the `@relation` declaration that is contained in the .arff file, as it is intended by the `MEKA project's dataset format <https://waikato.github.io/meka/datasets/>`_. According to the MEKA format, the number of labels may be specified by including the substring "-C L" in the `@relation` name, where "L" is the number of leading attributes in the dataset that should be treated as labels.
