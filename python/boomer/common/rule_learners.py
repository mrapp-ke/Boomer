#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
import os
from abc import abstractmethod
from ast import literal_eval
from enum import Enum
from typing import List

import numpy as np
from boomer.common.input_data import DenseLabelMatrix, DokLabelMatrix, DenseFeatureMatrix, CscFeatureMatrix
from boomer.common.prediction import Predictor
from boomer.common.pruning import Pruning, IREP
from boomer.common.rules import ModelBuilder
from boomer.common.sequential_rule_induction import SequentialRuleInduction
from boomer.common.stopping_criteria import StoppingCriterion, SizeStoppingCriterion, TimeStoppingCriterion
from boomer.common.sub_sampling import FeatureSubSampling, RandomFeatureSubsetSelection, InstanceSubSampling, Bagging, \
    RandomInstanceSubsetSelection, LabelSubSampling, RandomLabelSubsetSelection
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from sklearn.utils import check_array

from boomer.common.arrays import DTYPE_UINT8, DTYPE_UINT32, DTYPE_FLOAT32
from boomer.common.learners import Learner, NominalAttributeLearner

HEAD_REFINEMENT_SINGLE = 'single-label'

LABEL_SUB_SAMPLING_RANDOM = 'random-label-selection'

INSTANCE_SUB_SAMPLING_RANDOM = 'random-instance-selection'

INSTANCE_SUB_SAMPLING_BAGGING = 'bagging'

FEATURE_SUB_SAMPLING_RANDOM = 'random-feature-selection'

PRUNING_IREP = 'irep'

ARGUMENT_SAMPLE_SIZE = 'sample_size'

ARGUMENT_NUM_SAMPLES = 'num_samples'


class SparsePolicy(Enum):
    AUTO = 'auto'
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'


def create_sparse_policy(policy: str) -> SparsePolicy:
    try:
        return SparsePolicy(policy)
    except ValueError:
        raise ValueError('Invalid matrix format given: \'' + str(policy) + '\'. Must be one of ' + str(
            [x.value for x in SparsePolicy]))


def create_label_sub_sampling(label_sub_sampling: str, num_labels: int) -> LabelSubSampling:
    if label_sub_sampling is None:
        return None
    else:
        prefix, args = parse_prefix_and_dict(label_sub_sampling, [LABEL_SUB_SAMPLING_RANDOM])

        if prefix == LABEL_SUB_SAMPLING_RANDOM:
            num_samples = get_int_argument(args, ARGUMENT_NUM_SAMPLES, 1, lambda x: 1 <= x < num_labels)
            return RandomLabelSubsetSelection(num_samples)
        raise ValueError('Invalid value given for parameter \'label_sub_sampling\': ' + str(label_sub_sampling))


def create_instance_sub_sampling(instance_sub_sampling: str) -> InstanceSubSampling:
    if instance_sub_sampling is None:
        return None
    else:
        prefix, args = parse_prefix_and_dict(instance_sub_sampling,
                                             [INSTANCE_SUB_SAMPLING_BAGGING, INSTANCE_SUB_SAMPLING_RANDOM])

        if prefix == INSTANCE_SUB_SAMPLING_BAGGING:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 1.0, lambda x: 0 < x <= 1)
            return Bagging(sample_size)
        elif prefix == INSTANCE_SUB_SAMPLING_RANDOM:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.66, lambda x: 0 < x < 1)
            return RandomInstanceSubsetSelection(sample_size)
        raise ValueError('Invalid value given for parameter \'instance_sub_sampling\': ' + str(instance_sub_sampling))


def create_feature_sub_sampling(feature_sub_sampling: str) -> FeatureSubSampling:
    if feature_sub_sampling is None:
        return None
    else:
        prefix, args = parse_prefix_and_dict(feature_sub_sampling, [FEATURE_SUB_SAMPLING_RANDOM])

        if prefix == FEATURE_SUB_SAMPLING_RANDOM:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.0, lambda x: 0 <= x < 1)
            return RandomFeatureSubsetSelection(sample_size)
        raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))


def create_pruning(pruning: str) -> Pruning:
    if pruning is None:
        return None
    if pruning == PRUNING_IREP:
        return IREP()
    raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))


def create_stopping_criteria(max_rules: int, time_limit: int) -> List[StoppingCriterion]:
    stopping_criteria: List[StoppingCriterion] = []

    if max_rules != -1:
        if max_rules > 0:
            stopping_criteria.append(SizeStoppingCriterion(max_rules))
        else:
            raise ValueError('Invalid value given for parameter \'max_rules\': ' + str(max_rules))

    if time_limit != -1:
        if time_limit > 0:
            stopping_criteria.append(TimeStoppingCriterion(time_limit))
        else:
            raise ValueError('Invalid value given for parameter \'time_limit\': ' + str(time_limit))

    return stopping_criteria


def create_min_coverage(min_coverage: int) -> int:
    if min_coverage < 1:
        raise ValueError('Invalid value given for parameter \'min_coverage\': ' + str(min_coverage))

    return min_coverage


def create_max_conditions(max_conditions: int) -> int:
    if max_conditions != -1 and max_conditions < 1:
        raise ValueError('Invalid value given for parameter \'max_conditions\': ' + str(max_conditions))

    return max_conditions


def create_max_head_refinements(max_head_refinements: int) -> int:
    if max_head_refinements != -1 and max_head_refinements < 1:
        raise ValueError('Invalid value given for parameter \'max_head_refinements\': ' + str(max_head_refinements))

    return max_head_refinements


def create_num_threads(num_threads: int) -> int:
    if num_threads == -1:
        return os.cpu_count()
    elif num_threads < 1:
        raise ValueError('Invalid value given for parameter \'num_threads\': ' + str(num_threads))

    return num_threads


def parse_prefix_and_dict(string: str, prefixes: List[str]) -> [str, dict]:
    for prefix in prefixes:
        if string.startswith(prefix):
            suffix = string[len(prefix):].strip()

            if len(suffix) > 0:
                return prefix, literal_eval(suffix)

            return prefix, {}

    return None, None


def get_int_argument(args: dict, key: str, default: int, validation) -> int:
    if args is not None and key in args:
        value = int(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for int argument \'' + key + '\': ' + str(value))

        return value

    return default


def get_float_argument(args: dict, key: str, default: float, validation) -> float:
    if args is not None and key in args:
        value = float(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for float argument \'' + key + '\': ' + str(value))

        return value

    return default


def should_enforce_sparse(m, sparse_format: str, policy: SparsePolicy) -> bool:
    """
    Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix`,
    `scipy.sparse.csc_matrix` or `scipy.sparse.dok_matrix`, depending on the format of the given matrix and a given
    `SparsePolicy`:

    - If the given policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if possible,
    if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the matrix into a
    sparse format, it must be a `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix` or `scipy.sparse.coo_matrix`. If
    the given sparse format is `csr` or `csc` and the matrix is a already in that format, it will not be converted.

    - If the given policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into the specified sparse
    format, if possible.

    - If the given policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The sparse format to be used. Must be 'csr', 'csc', or `dok`
    :param policy:          The `SparsePolicy` to be used
    :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given format,
                            False otherwise
    """
    if sparse_format != 'csr' and sparse_format != 'csc' and sparse_format != 'dok':
        raise ValueError('Invalid sparse format given: ' + str(sparse_format))

    if not issparse(m):
        # Given matrix is dense
        if policy != SparsePolicy.FORCE_SPARSE:
            return False
    elif (isspmatrix_csr(m) and sparse_format == 'csr') or (isspmatrix_csc(m) and sparse_format == 'csc'):
        # Matrix is a `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix` and is already in the given sparse format
        return policy != SparsePolicy.FORCE_DENSE
    elif isspmatrix_lil(m) or isspmatrix_coo(m) or isspmatrix_dok(m):
        # Given matrix is in a format that might be converted into the specified sparse format
        if policy == SparsePolicy.AUTO:
            num_non_zero = m.nnz

            if sparse_format == 'dok':
                size_sparse = np.dtype(DTYPE_UINT32).itemsize * 2 * num_non_zero
                size_dense = np.prod(m.shape) * np.dtype(DTYPE_UINT8).itemsize
            else:
                num_pointers = m.shape[1 if sparse_format == 'csc' else 0]
                size_int = np.dtype(DTYPE_UINT32).itemsize
                size_float = np.dtype(DTYPE_FLOAT32).itemsize
                size_sparse = (num_non_zero * size_float) + (num_non_zero * size_int) + (num_pointers * size_int)
                size_dense = np.prod(m.shape) * size_float

            return size_sparse < size_dense
        else:
            return policy == SparsePolicy.FORCE_SPARSE

    raise ValueError(
        'Matrix of type ' + type(m).__name__ + ' cannot be converted to format \'' + str(sparse_format) + '\'')


class MLRuleLearner(Learner, NominalAttributeLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.

    Attributes
        num_labels_ The number of labels in the training data set
    """

    def __init__(self, random_state: int, feature_format: str, label_format: str):
        """
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        :param feature_format:  The format to be used for the feature matrix. Must be 'sparse', 'dense' or 'auto'
        :param label_format:    The format to be used for the label matrix. Must be 'sparse', 'dense' or 'auto'
        """
        super().__init__()
        self.random_state = random_state
        self.feature_format = feature_format
        self.label_format = label_format

    def _fit(self, x, y):
        # Validate feature matrix and convert it to the preferred format...
        x_sparse_format = 'csc'
        x_sparse_policy = create_sparse_policy(self.feature_format)
        x_enforce_sparse = should_enforce_sparse(x, sparse_format=x_sparse_format, policy=x_sparse_policy)
        x = self._validate_data((x if x_enforce_sparse else x.toarray(order='F')),
                                accept_sparse=(x_sparse_format if x_enforce_sparse else False), dtype=DTYPE_FLOAT32)

        if issparse(x):
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            feature_matrix = CscFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
        else:
            feature_matrix = DenseFeatureMatrix(x)

        # Validate label matrix and convert it to the preferred format...
        y_sparse_policy = create_sparse_policy(self.label_format)
        y_enforce_sparse = should_enforce_sparse(y, sparse_format='dok', policy=y_sparse_policy)
        y = check_array((y if y_enforce_sparse else y.toarray(order='C')),
                        accept_sparse=('lil' if y_enforce_sparse else False), ensure_2d=False, dtype=DTYPE_UINT8)

        if issparse(y):
            rows = np.ascontiguousarray(y.rows)
            label_matrix = DokLabelMatrix(y.shape[0], y.shape[1], rows)
        else:
            label_matrix = DenseLabelMatrix(y)

        num_labels = label_matrix.num_labels
        self.num_labels_ = num_labels

        # Create an array that contains the indices of all nominal attributes, if any...
        nominal_attribute_indices = self.nominal_attribute_indices

        if nominal_attribute_indices is not None and len(nominal_attribute_indices) > 0:
            nominal_attribute_mask = np.zeros((x.shape[1]), dtype=DTYPE_UINT8, order='C')
            nominal_attribute_mask[nominal_attribute_indices] = 1
        else:
            nominal_attribute_mask = None

        # Induce rules...
        sequential_rule_induction = self._create_sequential_rule_induction(num_labels)
        model_builder = self._create_model_builder()
        return sequential_rule_induction.induce_rules(nominal_attribute_mask, feature_matrix, label_matrix,
                                                      self.random_state, model_builder)

    def _predict(self, x):
        sparse_format = 'csr'
        sparse_policy = create_sparse_policy(self.feature_format)
        enforce_sparse = should_enforce_sparse(x, sparse_format=sparse_format, policy=sparse_policy)
        x = self._validate_data((x if enforce_sparse else x.toarray(order='C')), reset=False,
                                accept_sparse=(sparse_format if enforce_sparse else False), dtype=DTYPE_FLOAT32)
        num_labels = self.num_labels_
        model = self.model_
        predictor = self._create_predictor()

        if issparse(x):
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            num_features = x.shape[1]
            return predictor.predict_csr(x_data, x_row_indices, x_col_indices, num_features, num_labels, model)
        else:
            return predictor.predict(x, num_labels, model)

    @abstractmethod
    def _create_predictor(self) -> Predictor:
        """
        Must be implemented by subclasses in order to create the `Predictor` to be used for making predictions.

        :return: The `Predictor` that has been created
        """
        pass

    @abstractmethod
    def _create_sequential_rule_induction(self, num_labels: int) -> SequentialRuleInduction:
        """
        Must be implemented by subclasses in order to create the algorithm that should be used for sequential rule
        induction.

        :param num_labels:  The number of labels in the training data set
        :return:            The algorithm for sequential rule induction that has been created
        """
        pass

    def _create_model_builder(self) -> ModelBuilder:
        """
        Must be implemented by subclasses in order to create the builder that should be used for building the model.

        :return: The builder that has been created
        """
        pass
