#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
import logging as log
from abc import abstractmethod
from enum import Enum
from typing import Dict, Set, Optional

import numpy as np
from mlrl.common.arrays import enforce_dense
from mlrl.common.cython.feature_matrix import FortranContiguousFeatureMatrix, CscFeatureMatrix, CsrFeatureMatrix, \
    CContiguousFeatureMatrix
from mlrl.common.cython.label_matrix import CContiguousLabelMatrix, CsrLabelMatrix
from mlrl.common.cython.learner import RuleLearnerConfig, RuleLearner as RuleLearnerWrapper
from mlrl.common.cython.nominal_feature_mask import EqualNominalFeatureMask, MixedNominalFeatureMask
from mlrl.common.data_types import DTYPE_UINT8, DTYPE_UINT32, DTYPE_FLOAT32
from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.common.options import BooleanOption
from mlrl.common.options import Options
from mlrl.common.strings import format_enum_values, format_string_set, format_dict_keys
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from sklearn.utils import check_array

AUTOMATIC = 'auto'

NONE = 'none'

RULE_INDUCTION_TOP_DOWN = 'top-down'

ARGUMENT_USE_DEFAULT_RULE = 'default_rule'

RULE_MODEL_ASSEMBLAGE_SEQUENTIAL = 'sequential'

ARGUMENT_MIN_COVERAGE = 'min_coverage'

ARGUMENT_MAX_CONDITIONS = 'max_conditions'

ARGUMENT_MAX_HEAD_REFINEMENTS = 'max_head_refinements'

ARGUMENT_RECALCULATE_PREDICTIONS = 'recalculate_predictions'

SAMPLING_WITH_REPLACEMENT = 'with-replacement'

SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

SAMPLING_STRATIFIED_LABEL_WISE = 'stratified-label-wise'

SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

ARGUMENT_SAMPLE_SIZE = 'sample_size'

ARGUMENT_NUM_SAMPLES = 'num_samples'

PARTITION_SAMPLING_RANDOM = 'random'

ARGUMENT_HOLDOUT_SET_SIZE = 'holdout_set_size'

BINNING_EQUAL_FREQUENCY = 'equal-frequency'

BINNING_EQUAL_WIDTH = 'equal-width'

ARGUMENT_BIN_RATIO = 'bin_ratio'

ARGUMENT_MIN_BINS = 'min_bins'

ARGUMENT_MAX_BINS = 'max_bins'

PRUNING_IREP = 'irep'

ARGUMENT_NUM_THREADS = 'num_threads'

RULE_INDUCTION_VALUES: Dict[str, Set[str]] = {
    RULE_INDUCTION_TOP_DOWN: {ARGUMENT_MIN_COVERAGE, ARGUMENT_MAX_CONDITIONS, ARGUMENT_MAX_HEAD_REFINEMENTS,
                              ARGUMENT_RECALCULATE_PREDICTIONS}
}

RULE_MODEL_ASSEMBLAGE_VALUES: Dict[str, Set[str]] = {
    RULE_MODEL_ASSEMBLAGE_SEQUENTIAL: {ARGUMENT_USE_DEFAULT_RULE}
}

LABEL_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_NUM_SAMPLES}
}

FEATURE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE}
}

INSTANCE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    SAMPLING_WITH_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_SAMPLE_SIZE}
}

PARTITION_SAMPLING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    PARTITION_SAMPLING_RANDOM: {ARGUMENT_HOLDOUT_SET_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_HOLDOUT_SET_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_HOLDOUT_SET_SIZE}
}

FEATURE_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_FREQUENCY: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS}
}

PRUNING_VALUES: Set[str] = {
    NONE,
    PRUNING_IREP
}

PARALLEL_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_THREADS},
    str(BooleanOption.FALSE.value): {}
}


class SparsePolicy(Enum):
    AUTO = AUTOMATIC
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'


class SparseFormat(Enum):
    CSC = 'csc'
    CSR = 'csr'


def create_sparse_policy(parameter_name: str, policy: str) -> SparsePolicy:
    try:
        return SparsePolicy(policy)
    except ValueError:
        raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                         + format_enum_values(SparsePolicy) + ', but is "' + str(policy) + '"')


def configure_rule_model_assemblage(config: RuleLearnerConfig, rule_model_assemblage: Optional[str]):
    if rule_model_assemblage is not None:
        value, options = parse_param_and_options('rule_model_assemblage', rule_model_assemblage,
                                                 RULE_MODEL_ASSEMBLAGE_VALUES)

        if value == RULE_MODEL_ASSEMBLAGE_SEQUENTIAL:
            c = config.use_sequential_rule_model_assemblage()
            c.set_use_default_rule(options.get_bool(ARGUMENT_USE_DEFAULT_RULE, c.get_use_default_rule()))


def configure_rule_induction(config: RuleLearnerConfig, rule_induction: Optional[str]):
    if rule_induction is not None:
        value, options = parse_param_and_options('rule_induction', rule_induction, RULE_INDUCTION_VALUES)

        if value == RULE_INDUCTION_TOP_DOWN:
            c = config.use_top_down_rule_induction()
            c.set_min_coverage(options.get_int(ARGUMENT_MIN_COVERAGE, c.get_min_coverage()))
            c.set_max_conditions(options.get_int(ARGUMENT_MAX_CONDITIONS, c.get_max_conditions()))
            c.set_max_head_refinements(options.get_int(ARGUMENT_MAX_HEAD_REFINEMENTS, c.get_max_head_refinements()))
            c.set_recalculate_predictions(options.get_bool(ARGUMENT_RECALCULATE_PREDICTIONS,
                                                           c.get_recalculate_predictions()))


def configure_feature_binning(config: RuleLearnerConfig, feature_binning: Optional[str]):
    if feature_binning is not None:
        value, options = parse_param_and_options('feature_binning', feature_binning, FEATURE_BINNING_VALUES)

        if value == NONE:
            config.use_no_feature_binning()
        elif value == BINNING_EQUAL_FREQUENCY:
            c = config.use_equal_frequency_feature_binning()
            c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))
        elif value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_feature_binning()
            c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))


def configure_label_sampling(config: RuleLearnerConfig, label_sampling: Optional[str]):
    if label_sampling is not None:
        value, options = parse_param_and_options('label_sampling', label_sampling, LABEL_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_label_sampling()
        if value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_label_sampling_without_replacement()
            c.set_num_samples(options.get_int(ARGUMENT_NUM_SAMPLES, c.get_num_samples()))


def configure_instance_sampling(config: RuleLearnerConfig, instance_sampling: Optional[str]):
    if instance_sampling is not None:
        value, options = parse_param_and_options('instance_sampling', instance_sampling, INSTANCE_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_instance_sampling()
        elif value == SAMPLING_WITH_REPLACEMENT:
            c = config.use_instance_sampling_with_replacement()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_instance_sampling_without_replacement()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_STRATIFIED_LABEL_WISE:
            c = config.use_label_wise_stratified_instance_sampling()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            c = config.use_example_wise_stratified_instance_sampling()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))


def configure_feature_sampling(config: RuleLearnerConfig, feature_sampling: Optional[str]):
    if feature_sampling is not None:
        value, options = parse_param_and_options('feature_sampling', feature_sampling, FEATURE_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_feature_sampling()
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_feature_sampling_without_replacement()
            c.set_sample_size(options.get_float(ARGUMENT_SAMPLE_SIZE, c.get_sample_size()))


def configure_partition_sampling(config: RuleLearnerConfig, partition_sampling: Optional[str]):
    if partition_sampling is not None:
        value, options = parse_param_and_options('holdout', partition_sampling, PARTITION_SAMPLING_VALUES)

        if value == NONE:
            config.use_no_partition_sampling()
        elif value == PARTITION_SAMPLING_RANDOM:
            c = config.use_random_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_LABEL_WISE:
            c = config.use_label_wise_stratified_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            c = config.use_example_wise_stratified_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))


def configure_pruning(config: RuleLearnerConfig, pruning: Optional[str]):
    if pruning is not None:
        value = parse_param('pruning', pruning, PRUNING_VALUES)

        if value == NONE:
            config.use_no_pruning()
        elif value == PRUNING_IREP:
            config.use_irep_pruning()


def configure_parallel_rule_refinement(config: RuleLearnerConfig, parallel_rule_refinement: Optional[str]):
    if parallel_rule_refinement is not None:
        value, options = parse_param_and_options('parallel_rule_refinement', parallel_rule_refinement, PARALLEL_VALUES)

        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_rule_refinement()
        else:
            c = config.use_parallel_rule_refinement()
            c.set_num_threads(options.get_int(ARGUMENT_NUM_THREADS, c.get_num_threads()))


def configure_parallel_statistic_update(config: RuleLearnerConfig, parallel_statistic_update: Optional[str]):
    if parallel_statistic_update is not None:
        value, options = parse_param_and_options('parallel_statistic_update', parallel_statistic_update,
                                                 PARALLEL_VALUES)

        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_statistic_update()
        else:
            c = config.use_parallel_statistic_update()
            c.set_num_threads(options.get_int(ARGUMENT_NUM_THREADS, c.get_num_threads()))


def configure_parallel_prediction(config: RuleLearnerConfig, parallel_prediction: Optional[str]):
    if parallel_prediction is not None:
        value, options = parse_param_and_options('parallel_prediction', parallel_prediction, PARALLEL_VALUES)

        if value == BooleanOption.TRUE.value:
            c = config.use_parallel_prediction()
            c.set_num_threads(options.get_int(ARGUMENT_NUM_THREADS, c.get_num_threads()))
        else:
            config.use_no_parallel_prediction()


def configure_size_stopping_criterion(config: RuleLearnerConfig, max_rules: Optional[int]):
    if max_rules is not None:
        if max_rules == 0:
            config.use_no_size_stopping_criterion()
        else:
            config.use_size_stopping_criterion().set_max_rules(max_rules)


def configure_time_stopping_criterion(config: RuleLearnerConfig, time_limit: Optional[int]):
    if time_limit is not None:
        if time_limit == 0:
            config.use_no_time_stopping_criterion()
        else:
            config.use_time_stopping_criterion().set_time_limit(time_limit)


def parse_param(parameter_name: str, value: str, allowed_values: Set[str]) -> str:
    if value in allowed_values:
        return value

    raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                     + format_string_set(allowed_values) + ', but is "' + value + '"')


def parse_param_and_options(parameter_name: str, value: str,
                            allowed_values_and_options: Dict[str, Set[str]]) -> (str, Options):
    for allowed_value, allowed_options in allowed_values_and_options.items():
        if value.startswith(allowed_value):
            suffix = value[len(allowed_value):].strip()

            if len(suffix) > 0:
                try:
                    return allowed_value, Options.create(suffix, allowed_options)
                except ValueError as e:
                    raise ValueError('Invalid options specified for parameter "' + parameter_name + '" with value "'
                                     + allowed_value + '": ' + str(e))

            return allowed_value, Options()

    raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                     + format_dict_keys(allowed_values_and_options) + ', but is "' + value + '"')


def is_sparse(m, sparse_format: SparseFormat, dtype, sparse_values: bool = True) -> bool:
    """
    Returns whether a given matrix is considered sparse or not. A matrix is considered sparse if it is given in a sparse
    format and is expected to occupy less memory than a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param dtype:           The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if the given matrix is considered sparse, False otherwise
    """
    if issparse(m):
        num_pointers = m.shape[1 if sparse_format == SparseFormat.CSC else 0]
        size_int = np.dtype(DTYPE_UINT32).itemsize
        size_data = np.dtype(dtype).itemsize if sparse_values else 0
        num_non_zero = m.nnz
        size_sparse = (num_non_zero * size_data) + (num_non_zero * size_int) + (num_pointers * size_int)
        size_dense = np.prod(m.shape) * size_data
        return size_sparse < size_dense
    return False


def should_enforce_sparse(m, sparse_format: SparseFormat, policy: SparsePolicy, dtype,
                          sparse_values: bool = True) -> bool:
    """
    Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix`,
    `scipy.sparse.csc_matrix` or `scipy.sparse.dok_matrix`, depending on the format of the given matrix and a given
    `SparsePolicy`:

    If the given policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if possible,
    if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the matrix into a
    sparse format, it must be a `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix` or `scipy.sparse.coo_matrix`. If
    the given sparse format is `csr` or `csc` and the matrix is a already in that format, it will not be converted.

    If the given policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into the specified sparse
    format, if possible.

    If the given policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param policy:          The `SparsePolicy` to be used
    :param dtype:           The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given format,
                            False otherwise
    """
    if not issparse(m):
        # Given matrix is dense
        if policy != SparsePolicy.FORCE_SPARSE:
            return False
    elif (isspmatrix_csr(m) and sparse_format == SparseFormat.CSR) or (
            isspmatrix_csc(m) and sparse_format == SparseFormat.CSC):
        # Matrix is a `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix` and is already in the given sparse format
        return policy != SparsePolicy.FORCE_DENSE
    elif isspmatrix_lil(m) or isspmatrix_coo(m) or isspmatrix_dok(m):
        # Given matrix is in a format that might be converted into the specified sparse format
        if policy == SparsePolicy.AUTO:
            return is_sparse(m, sparse_format=sparse_format, dtype=dtype, sparse_values=sparse_values)
        else:
            return policy == SparsePolicy.FORCE_SPARSE

    raise ValueError(
        'Matrix of type ' + type(m).__name__ + ' cannot be converted to format "' + str(sparse_format) + '""')


class MLRuleLearner(Learner, NominalAttributeLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.
    """

    def __init__(self, random_state: int, feature_format: str, label_format: str, prediction_format: str):
        """
        :param random_state:        The seed to be used by RNGs. Must be at least 1
        :param feature_format:      The format to be used for the representation of the feature matrix. Must be
                                    `sparse`, `dense` or `auto`
        :param label_format:        The format to be used for the representation of the label matrix. Must be `sparse`,
                                    `dense` or 'auto'
        :param prediction_format:   The format to be used for representation of predicted labels. Must be `sparse`,
                                    `dense` or `auto`
        """
        super().__init__()
        self.random_state = random_state
        self.feature_format = feature_format
        self.label_format = label_format
        self.prediction_format = prediction_format

    def _fit(self, x, y):
        # Validate feature matrix and convert it to the preferred format...
        x_sparse_format = SparseFormat.CSC
        x_sparse_policy = create_sparse_policy('feature_format', self.feature_format)
        x_enforce_sparse = should_enforce_sparse(x, sparse_format=x_sparse_format, policy=x_sparse_policy,
                                                 dtype=DTYPE_FLOAT32)
        x = self._validate_data((x if x_enforce_sparse else enforce_dense(x, order='F', dtype=DTYPE_FLOAT32)),
                                accept_sparse=(x_sparse_format.value if x_enforce_sparse else False),
                                dtype=DTYPE_FLOAT32, force_all_finite='allow-nan')

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the training examples')
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            feature_matrix = CscFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
        else:
            log.debug('A dense matrix is used to store the feature values of the training examples')
            feature_matrix = FortranContiguousFeatureMatrix(x)

        # Validate label matrix and convert it to the preferred format...
        y_sparse_format = SparseFormat.CSR

        # Check if predictions should be sparse...
        prediction_sparse_policy = create_sparse_policy('prediction_format', self.prediction_format)
        self.sparse_predictions_ = prediction_sparse_policy != SparsePolicy.FORCE_DENSE and (
                prediction_sparse_policy == SparsePolicy.FORCE_SPARSE or
                is_sparse(y, sparse_format=y_sparse_format, dtype=DTYPE_UINT8, sparse_values=True))

        y_sparse_policy = create_sparse_policy('label_format', self.label_format)
        y_enforce_sparse = should_enforce_sparse(y, sparse_format=y_sparse_format, policy=y_sparse_policy,
                                                 dtype=DTYPE_UINT8, sparse_values=False)
        y = check_array((y if y_enforce_sparse else enforce_dense(y, order='C', dtype=DTYPE_UINT8)),
                        accept_sparse=(y_sparse_format.value if y_enforce_sparse else False), ensure_2d=False,
                        dtype=DTYPE_UINT8)

        if issparse(y):
            log.debug('A sparse matrix is used to store the labels of the training examples')
            y_row_indices = np.ascontiguousarray(y.indptr, dtype=DTYPE_UINT32)
            y_col_indices = np.ascontiguousarray(y.indices, dtype=DTYPE_UINT32)
            label_matrix = CsrLabelMatrix(y.shape[0], y.shape[1], y_row_indices, y_col_indices)
        else:
            log.debug('A dense matrix is used to store the labels of the training examples')
            label_matrix = CContiguousLabelMatrix(y)

        # Create a mask that provides access to the information whether individual features are nominal or not...
        num_features = feature_matrix.get_num_cols()

        if self.nominal_attribute_indices is None or len(self.nominal_attribute_indices) == 0:
            nominal_feature_mask = EqualNominalFeatureMask(False)
        elif len(self.nominal_attribute_indices) == num_features:
            nominal_feature_mask = EqualNominalFeatureMask(True)
        else:
            nominal_feature_mask = MixedNominalFeatureMask(num_features, self.nominal_attribute_indices)

        # Induce rules...
        learner = self._create_learner()
        training_result = learner.fit(nominal_feature_mask, feature_matrix, label_matrix, self.random_state)
        self.num_labels_ = training_result.num_labels
        self.label_space_info_ = training_result.label_space_info
        return training_result.rule_model

    def _predict(self, x):
        feature_matrix = self.__create_row_wise_feature_matrix(x)
        learner = self._create_learner()

        if self.sparse_predictions_:
            log.debug('A sparse matrix is used to store the predicted labels')
            return learner.predict_sparse_labels(feature_matrix, self.model_, self.label_space_info_, self.num_labels_)
        else:
            log.debug('A dense matrix is used to store the predicted labels')
            return learner.predict_labels(feature_matrix, self.model_, self.label_space_info_, self.num_labels_)

    def _predict_proba(self, x):
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x)
        num_labels = self.num_labels_

        if learner.can_predict_probabilities(feature_matrix, num_labels):
            log.debug('A dense matrix is used to store the predicted probability estimates')
            return learner.predict_probabilities(feature_matrix, self.model_, self.label_space_info_, num_labels)
        else:
            super()._predict_proba(x)

    def __create_row_wise_feature_matrix(self, x):
        sparse_format = SparseFormat.CSR
        sparse_policy = create_sparse_policy('feature_format', self.feature_format)
        enforce_sparse = should_enforce_sparse(x, sparse_format=sparse_format, policy=sparse_policy,
                                               dtype=DTYPE_FLOAT32)
        x = self._validate_data(x if enforce_sparse else enforce_dense(x, order='C', dtype=DTYPE_FLOAT32), reset=False,
                                accept_sparse=(sparse_format.value if enforce_sparse else False), dtype=DTYPE_FLOAT32,
                                force_all_finite='allow-nan')

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the query examples')
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            return CsrFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
        else:
            log.debug('A dense matrix is used to store the feature values of the query examples')
            return CContiguousFeatureMatrix(x)

    @abstractmethod
    def _create_learner(self) -> RuleLearnerWrapper:
        """
        Must be implemented by subclasses in order to configure and create an implementation of the rule learner.

        :return: The implementation of the rule learner that has been created
        """
        pass
