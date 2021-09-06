Running the Algorithm
---------------------

The Python script ``python/main_boomer.py`` allows to run experiments on a specific data set using different configurations of the learning algorithm. Besides the training and evaluation of models, the script does also allow to write experimental results into an output directory. Furthermore, the learned models can be stored on disk for later use.

In the following, an example of how the script can be executed is shown:

.. code-block:: text

   venv/bin/python3 python/main_boomer.py --data-dir /path/to/data --output-dir /path/to/results/emotions --model-dir /path/to/models/emotions --dataset emotions --folds 10 --max-rules 1000 --instance-sampling with-replacement --feature-sampling without-replacement --loss logistic-label-wise --shrinkage 0.3 --pruning None --head-type single-label

