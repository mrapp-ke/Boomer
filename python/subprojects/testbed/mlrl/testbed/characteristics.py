"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions to determine certain characteristics of feature or label matrices.
"""
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np

from scipy.sparse import issparse

from mlrl.common.options import Options

from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE, Formatter, filter_formatters, format_table
from mlrl.testbed.output_writer import Formattable, Tabularizable

OPTION_LABELS = 'labels'

OPTION_LABEL_DENSITY = 'label_density'

OPTION_LABEL_SPARSITY = 'label_sparsity'

OPTION_LABEL_IMBALANCE_RATIO = 'label_imbalance_ratio'

OPTION_LABEL_CARDINALITY = 'label_cardinality'

OPTION_DISTINCT_LABEL_VECTORS = 'distinct_label_vectors'


def density(m) -> float:
    """
    Calculates and returns the density of a given feature or label matrix.

    :param m:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_rows, num_cols)`, that stores the feature values
                of training examples or their labels
    :return:    The fraction of non-zero elements in the given matrix among all elements
    """
    num_elements = m.shape[0] * m.shape[1]

    if issparse(m):
        num_non_zero = m.nnz
    else:
        num_non_zero = np.count_nonzero(m)

    return num_non_zero / num_elements if num_elements > 0 else 0


def label_cardinality(y) -> float:
    """
    Calculates and returns the average label cardinality of a given label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The average number of relevant labels per training example
    """
    if issparse(y):
        y = y.tolil()
        num_relevant_per_example = y.getnnz(axis=1)
    else:
        num_relevant_per_example = np.count_nonzero(y, axis=1)

    return np.average(num_relevant_per_example)


def distinct_label_vectors(y) -> int:
    """
    Determines and returns the number of distinct label vectors in a label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The number of distinct label vectors in the given matrix
    """
    if issparse(y):
        y = y.tolil()
        return np.unique(y.rows).shape[0]
    else:
        return np.unique(y, axis=0).shape[0]


def label_imbalance_ratio(y) -> float:
    """
    Calculates and returns the average label imbalance ratio of a given label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The label imbalance ratio averaged over the available labels
    """
    if issparse(y):
        y = y.tocsc()
        num_relevant_per_label = y.getnnz(axis=0)
    else:
        num_relevant_per_label = np.count_nonzero(y, axis=0)

    num_relevant_per_label = num_relevant_per_label[num_relevant_per_label != 0]

    if num_relevant_per_label.shape[0] > 0:
        return np.average(np.max(num_relevant_per_label) / num_relevant_per_label)
    else:
        return 0.0


class LabelCharacteristics(Formattable, Tabularizable):
    """
    Stores characteristics of a label matrix.
    """

    def __init__(self, y):
        """
        :param y: A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
        """
        self._y = y
        self.num_labels = y.shape[1]

    @cached_property
    def label_density(self):
        return density(self._y)

    @property
    def label_sparsity(self):
        return 1 - self.label_density

    @cached_property
    def avg_label_imbalance_ratio(self):
        return label_imbalance_ratio(self._y)

    @cached_property
    def avg_label_cardinality(self):
        return label_cardinality(self._y)

    @cached_property
    def num_distinct_label_vectors(self):
        return distinct_label_vectors(self._y)

    def format(self, options: Options, **kwargs) -> str:
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 2)
        rows = []

        for formatter in filter_formatters(LABEL_CHARACTERISTICS, [options]):
            rows.append([formatter.name, formatter.format(self, percentage=percentage, decimals=decimals)])

        return format_table(rows)

    def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
        percentage = options.get_bool(OPTION_PERCENTAGE, True)
        decimals = options.get_int(OPTION_DECIMALS, 0)
        columns = {}

        for formatter in filter_formatters(LABEL_CHARACTERISTICS, [options]):
            columns[formatter] = formatter.format(self, percentage=percentage, decimals=decimals)

        return [columns]


class Characteristic(Formatter):
    """
    Allows to create textual representations of characteristics.
    """

    def __init__(self, option: str, name: str, getter_function, percentage: bool = False):
        """
        :param getter_function: The getter function that should be used to retrieve the characteristic
        """
        super().__init__(option, name, percentage)
        self.getter_function = getter_function

    def format(self, value, **kwargs) -> str:
        return super().format(self.getter_function(value), **kwargs)


LABEL_CHARACTERISTICS: List[Characteristic] = [
    Characteristic(OPTION_LABELS, 'Labels', lambda x: x.num_labels),
    Characteristic(OPTION_LABEL_DENSITY, 'Label Density', lambda x: x.label_density, percentage=True),
    Characteristic(OPTION_LABEL_SPARSITY, 'Label Sparsity', lambda x: x.label_sparsity, percentage=True),
    Characteristic(OPTION_LABEL_IMBALANCE_RATIO, 'Label Imbalance Ratio', lambda x: x.avg_label_imbalance_ratio),
    Characteristic(OPTION_LABEL_CARDINALITY, 'Label Cardinality', lambda x: x.avg_label_cardinality),
    Characteristic(OPTION_DISTINCT_LABEL_VECTORS, 'Distinct Label Vectors', lambda x: x.num_distinct_label_vectors)
]
