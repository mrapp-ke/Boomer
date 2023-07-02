"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path

from test_common import DATASET_EMOTIONS, DIR_DATA, DIR_OUT, HOLDOUT_NO, HOLDOUT_RANDOM, \
    HOLDOUT_STRATIFIED_EXAMPLE_WISE, HOLDOUT_STRATIFIED_LABEL_WISE, PREDICTION_TYPE_PROBABILITIES, \
    PREDICTION_TYPE_SCORES, CmdBuilder, CommonIntegrationTests, SkipTestOnCI

CMD_BOOMER = 'boomer'

FEATURE_BINNING_EQUAL_WIDTH = 'equal-width'

FEATURE_BINNING_EQUAL_FREQUENCY = 'equal-frequency'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

LOSS_SQUARED_HINGE_EXAMPLE_WISE = 'squared-hinge-example-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_ERROR_EXAMPLE_WISE = 'squared-error-example-wise'

HEAD_TYPE_SINGLE_LABEL = 'single-label'

HEAD_TYPE_COMPLETE = 'complete'

HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

LABEL_BINNING_NO = 'none'

LABEL_BINNING_EQUAL_WIDTH = 'equal-width'

PROBABILITY_CALIBRATOR_ISOTONIC = 'isotonic'

BINARY_PREDICTOR_AUTO = 'auto'

BINARY_PREDICTOR_LABEL_WISE = 'label-wise'

BINARY_PREDICTOR_LABEL_WISE_BASED_ON_PROBABILITIES = BINARY_PREDICTOR_LABEL_WISE + '{based_on_probabilities=true}'

BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES = BINARY_PREDICTOR_EXAMPLE_WISE + '{based_on_probabilities=true}'

BINARY_PREDICTOR_GFM = 'gfm'

PROBABILITY_PREDICTOR_AUTO = 'auto'

PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

GLOBAL_PRUNING_PRE = 'pre-pruning'

GLOBAL_PRUNING_POST = 'post-pruning'


class BoostingCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for running the BOOMER algorithm.
    """

    def __init__(self, data_dir: str = DIR_DATA, dataset: str = DATASET_EMOTIONS):
        super(BoostingCmdBuilder, self).__init__(cmd=CMD_BOOMER, data_dir=data_dir, dataset=dataset)

    def feature_binning(self, feature_binning: str = FEATURE_BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for feature binning.

        :param feature_binning: The name of the method that should be used for feature binning
        :return:                The builder itself
        """
        self.args.append('--feature-binning')
        self.args.append(feature_binning)
        return self

    def loss(self, loss: str = LOSS_LOGISTIC_LABEL_WISE):
        """
        Configures the algorithm to use a specific loss function.

        :param loss:    The name of the loss function that should be used
        :return:        The builder itself
        """
        self.args.append('--loss')
        self.args.append(loss)
        return self

    def marginal_probability_calibration(self, probability_calibrator: str = PROBABILITY_CALIBRATOR_ISOTONIC):
        """
        Configures the algorithm to fit a model for the calibration of marginal probabilities.

        :param probability_calibrator:  The name of the method that should be used to fit a calibration model
        :return:                        The builder itself
        """
        self.args.append('--marginal-probability-calibration')
        self.args.append(probability_calibrator)
        return self

    def joint_probability_calibration(self, probability_calibrator: str = PROBABILITY_CALIBRATOR_ISOTONIC):
        """
        Configures the algorithm to fit a model for the calibration of joint probabilities.

        :param probability_calibrator:  The name of the method that should be used to fit a calibration model
        :return:                        The builder itself
        """
        self.args.append('--joint-probability-calibration')
        self.args.append(probability_calibrator)
        return self

    def binary_predictor(self, binary_predictor: str = BINARY_PREDICTOR_AUTO):
        """
        Configures the algorithm to use a specific method for predicting binary labels.

        :param binary_predictor:    The name of the method that should be used for predicting binary labels
        :return:                    The builder itself
        """
        self.args.append('--binary-predictor')
        self.args.append(binary_predictor)
        return self

    def probability_predictor(self, probability_predictor: str = PROBABILITY_PREDICTOR_AUTO):
        """
        Configures the algorithm to use a specific method for predicting probabilities.

        :param probability_predictor:   The name of the method that should be used for predicting probabilities
        :return:                        The builder itself
        """
        self.args.append('--probability-predictor')
        self.args.append(probability_predictor)
        return self

    def default_rule(self, default_rule: bool = True):
        """
        Configures whether the algorithm should induce a default rule or not.

        :param default_rule:    True, if a default rule should be induced, False otherwise
        :return:                The builder itself
        """
        self.args.append('--default-rule')
        self.args.append(str(default_rule).lower())
        return self

    def head_type(self, head_type: str = HEAD_TYPE_SINGLE_LABEL):
        """
        Configures the algorithm to use a specific type of rule heads.

        :param head_type:   The type of rule heads to be used
        :return:            The builder itself
        """
        self.args.append('--head-type')
        self.args.append(head_type)
        return self

    def label_binning(self, label_binning: str = LABEL_BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for the assignment of labels to bins.

        :param label_binning:   The name of the method to be used
        :return:                The builder itself
        """
        self.args.append('--label-binning')
        self.args.append(label_binning)
        return self

    def sparse_statistic_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to store the statistics or not.

        :param sparse:  True, if sparse data structures should be used to store the statistics, False otherwise
        :return:        The builder itself
        """
        self.args.append('--statistic-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def global_pruning(self, global_pruning: str = GLOBAL_PRUNING_POST):
        """
        Configures the algorithm to use a specific method for pruning entire rules.

        :param global_pruning:  The name of the method that should be used for pruning entire rules
        :return:                The builder itself
        """
        self.args.append('--global-pruning')
        self.args.append(global_pruning)
        return self


class BoostingIntegrationTests(CommonIntegrationTests):
    """
    Defines a series of integration tests for the BOOMER algorithm.
    """

    def __init__(self, methodName='runTest'):
        """
        :param methodName: The name of the test method to be executed
        """
        super(BoostingIntegrationTests, self).__init__(cmd=CMD_BOOMER,
                                                       expected_output_dir=path.join(DIR_OUT, CMD_BOOMER),
                                                       methodName=methodName)

    def test_single_label_regression(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting regression scores for a single-label
        problem.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_single_label) \
            .prediction_type(PREDICTION_TYPE_SCORES) \
            .print_evaluation()
        self.run_cmd(builder, 'single-label-regression')

    def test_single_label_probabilities(self):
        """
        Tests the evaluation of the rule learning algorithm when predicting probabilities for a single-label problem.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_single_label) \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .print_evaluation()
        self.run_cmd(builder, 'single-label-probabilities')

    def test_feature_binning_equal_width_binary_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with binary
        features using a dense feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'feature-binning-equal-width_binary-features-dense')

    def test_feature_binning_equal_width_binary_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with binary
        features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        self.run_cmd(builder, 'feature-binning-equal-width_binary-features-sparse')

    def test_feature_binning_equal_width_nominal_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with nominal
        features using a dense feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'feature-binning-equal-width_nominal-features-dense')

    def test_feature_binning_equal_width_nominal_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with nominal
        features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        self.run_cmd(builder, 'feature-binning-equal-width_nominal-features-sparse')

    def test_feature_binning_equal_width_numerical_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with numerical
        features using a dense feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'feature-binning-equal-width_numerical-features-dense')

    def test_feature_binning_equal_width_numerical_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-width feature binning when applied to a dataset with numerical
        features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_WIDTH) \
            .sparse_feature_format()
        self.run_cmd(builder, 'feature-binning-equal-width_numerical-features-sparse')

    def test_feature_binning_equal_frequency_binary_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        binary features using a dense feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'feature-binning-equal-frequency_binary-features-dense')

    def test_feature_binning_equal_frequency_binary_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        binary features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_binary) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        self.run_cmd(builder, 'feature-binning-equal-frequency_binary-features-sparse')

    def test_feature_binning_equal_frequency_nominal_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        nominal features using a dense feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'feature-binning-equal-frequency_nominal-features-dense')

    def test_feature_binning_equal_frequency_nominal_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        nominal features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_nominal) \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        self.run_cmd(builder, 'feature-binning-equal-frequency_nominal-features-sparse')

    def test_feature_binning_equal_frequency_numerical_features_dense(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        numerical features using a dense feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format(False)
        self.run_cmd(builder, 'feature-binning-equal-frequency_numerical-features-dense')

    def test_feature_binning_equal_frequency_numerical_features_sparse(self):
        """
        Tests the BOOMER algorithm's ability to use equal-frequency feature binning when applied to a dataset with
        numerical features using a sparse feature representation.
        """
        builder = BoostingCmdBuilder() \
            .feature_binning(FEATURE_BINNING_EQUAL_FREQUENCY) \
            .sparse_feature_format()
        self.run_cmd(builder, 'feature-binning-equal-frequency_numerical-features-sparse')

    def test_loss_logistic_label_wise(self):
        """
        Tests the BOOMER algorithm when using the label-wise logistic loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE)
        self.run_cmd(builder, 'loss-logistic-label-wise')

    @SkipTestOnCI
    def test_loss_logistic_example_wise(self):
        """
        Tests the BOOMER algorithm when using the example-wise logistic loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE)
        self.run_cmd(builder, 'loss-logistic-example-wise')

    def test_loss_squared_hinge_label_wise(self):
        """
        Tests the BOOMER algorithm when using the label-wise squared hinge loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_HINGE_LABEL_WISE)
        self.run_cmd(builder, 'loss-squared-hinge-label-wise')

    @SkipTestOnCI
    def test_loss_squared_hinge_example_wise(self):
        """
        Tests the BOOMER algorithm when using the example-wise squared hinge loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_HINGE_EXAMPLE_WISE)
        self.run_cmd(builder, 'loss-squared-hinge-example-wise')

    def test_loss_squared_error_label_wise(self):
        """
        Tests the BOOMER algorithm when using the label-wise squared error loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_ERROR_LABEL_WISE)
        self.run_cmd(builder, 'loss-squared-error-label-wise')

    @SkipTestOnCI
    def test_loss_squared_error_example_wise(self):
        """
        Tests the BOOMER algorithm when using the example-wise squared error loss function.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_SQUARED_ERROR_EXAMPLE_WISE)
        self.run_cmd(builder, 'loss-squared-error-example-wise')

    def test_predictor_binary_label_wise(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained for each label individually.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_LABEL_WISE) \
            .print_predictions()
        self.run_cmd(builder, 'predictor-binary-label-wise')

    def test_predictor_binary_label_wise_based_on_probabilities(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained for each label individually based on
        probability estimates.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_LABEL_WISE_BASED_ON_PROBABILITIES) \
            .print_predictions() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-label-wise_based-on-probabilities')

    def test_predictor_binary_label_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting binary labels
        that are obtained for each label individually.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_LABEL_WISE) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'predictor-binary-label-wise_incremental')

    def test_predictor_binary_label_wise_incremental_based_on_probabilities(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting binary labels
        that are obtained for each label individually based on probability estimates.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_LABEL_WISE_BASED_ON_PROBABILITIES) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-label-wise_incremental_based-on-probabilities')

    def test_predictor_binary_label_wise_sparse(self):
        """
        Tests the BOOMER algorithm when predicting sparse binary labels that are obtained for each label individually.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_LABEL_WISE) \
            .print_predictions() \
            .sparse_prediction_format()
        self.run_cmd(builder, 'predictor-binary-label-wise_sparse')

    def test_predictor_binary_label_wise_sparse_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting sparse binary
        labels that are obtained for each label individually.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_LABEL_WISE) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'predictor-binary-label-wise_sparse_incremental')

    def test_predictor_binary_example_wise(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained by predicting one of the known label
        vectors.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_label_vectors()
        self.run_cmd(builder, 'predictor-binary-example-wise')

    def test_predictor_binary_example_wise_based_on_probabilities(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained by predicting one of the known label
        vectors based on probability estimates.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES) \
            .print_predictions() \
            .print_label_vectors() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-example-wise_based-on-probabilities')

    def test_predictor_binary_example_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting one of the
        known label vectors.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'predictor-binary-example-wise_incremental')

    def test_predictor_binary_example_wise_incremental_based_on_probabilities(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting one of the
        known label vectors based on probability estimates.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE_BASED_ON_PROBABILITIES) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-example-wise_incremental_based-on-probabilities')

    def test_predictor_binary_example_wise_sparse(self):
        """
        Tests the BOOMER algorithm when predicting sparse binary labels that are obtained by predicting one of the known
        label vectors.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .print_predictions() \
            .print_label_vectors() \
            .sparse_prediction_format()
        self.run_cmd(builder, 'predictor-binary-example-wise_sparse')

    def test_predictor_binary_example_wise_sparse_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting one of the
        known label vectors.
        """
        builder = BoostingCmdBuilder() \
            .binary_predictor(BINARY_PREDICTOR_EXAMPLE_WISE) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'predictor-binary-example-wise_sparse_incremental')

    def test_predictor_binary_gfm(self):
        """
        Tests the BOOMER algorithm when predicting binary labels that are obtained via the general F-measure maximizer
        (GFM).
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_label_vectors() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-gfm')

    def test_predictor_binary_gfm_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting binary labels
        that are obtained via the general F-measure maximizer (GFM).
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-gfm_incremental')

    def test_predictor_binary_gfm_sparse(self):
        """
        Tests the BOOMER algorithm when predicting sparse binary labels that are obtained via the general F-measure
        maximizer (GFM).
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .print_predictions() \
            .print_label_vectors() \
            .sparse_prediction_format() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-gfm_sparse')

    def test_predictor_binary_gfm_sparse_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting sparse binary
        labels that are obtained via the general F-measure maximizer (GFM).
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .binary_predictor(BINARY_PREDICTOR_GFM) \
            .sparse_prediction_format() \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-binary-gfm_sparse_incremental')

    def test_predictor_score_label_wise(self):
        """
        Tests the BOOMER algorithm when predicting regression scores that are obtained in label-wise manner.
        """
        builder = BoostingCmdBuilder() \
            .prediction_type(PREDICTION_TYPE_SCORES) \
            .print_predictions()
        self.run_cmd(builder, 'predictor-score-label-wise')

    def test_predictor_score_label_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting regression
        scores that are obtained in a label-wise manner.
        """
        builder = BoostingCmdBuilder() \
            .prediction_type(PREDICTION_TYPE_SCORES) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation()
        self.run_cmd(builder, 'predictor-score-label-wise_incremental')

    def test_predictor_probability_label_wise(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained by applying a label-wise
        transformation function.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_LABEL_WISE) \
            .print_predictions() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-probability-label-wise')

    def test_predictor_probability_label_wise_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting probabilities
        that are obtained by applying a label-wise transformation function.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_LABEL_WISE) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-probability-label-wise_incremental')

    def test_predictor_probability_marginalized(self):
        """
        Tests the BOOMER algorithm when predicting probabilities that are obtained via marginalization over the known
        label vectors.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .set_output_dir() \
            .store_evaluation(False) \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_MARGINALIZED) \
            .print_predictions() \
            .print_label_vectors() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-probability-marginalized')

    def test_predictor_probability_marginalized_incremental(self):
        """
        Tests the repeated evaluation of a model that is learned by the BOOMER algorithm when predicting probabilities
        that are obtained via marginalization over the known label vectors.
        """
        builder = BoostingCmdBuilder() \
            .marginal_probability_calibration() \
            .print_marginal_probability_calibration_model() \
            .store_marginal_probability_calibration_model() \
            .joint_probability_calibration() \
            .print_joint_probability_calibration_model() \
            .store_joint_probability_calibration_model() \
            .prediction_type(PREDICTION_TYPE_PROBABILITIES) \
            .probability_predictor(PROBABILITY_PREDICTOR_MARGINALIZED) \
            .incremental_evaluation() \
            .set_output_dir() \
            .print_evaluation() \
            .store_evaluation() \
            .set_model_dir()
        self.run_cmd(builder, 'predictor-probability-marginalized_incremental')

    def test_no_default_rule(self):
        """
        Tests the BOOMER algorithm when not inducing a default rule.
        """
        builder = BoostingCmdBuilder() \
            .default_rule(False) \
            .print_model_characteristics()
        self.run_cmd(builder, 'no-default-rule')

    def test_statistics_sparse_label_format_dense(self):
        """
        Tests the BOOMER algorithm when using sparse data structures for storing the statistics and a dense label
        representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_numerical) \
            .sparse_statistic_format() \
            .sparse_label_format(False) \
            .default_rule(False) \
            .loss(LOSS_SQUARED_HINGE_LABEL_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL)
        self.run_cmd(builder, 'statistics-sparse_label-format-dense')

    def test_statistics_sparse_label_format_sparse(self):
        """
        Tests the BOOMER algorithm when using sparse data structures for storing the statistics and a sparse label
        representation.
        """
        builder = BoostingCmdBuilder(dataset=self.dataset_numerical) \
            .sparse_statistic_format() \
            .sparse_label_format() \
            .default_rule(False) \
            .loss(LOSS_SQUARED_HINGE_LABEL_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL)
        self.run_cmd(builder, 'statistics-sparse_label-format-sparse')

    def test_label_wise_single_label_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules with
        single-label heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-single-label-heads')

    def test_label_wise_complete_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules with
        complete heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-complete-heads')

    def test_label_wise_complete_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function and equal-width label binning for
        the induction of rules with complete heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-complete-heads_equal-width-label-binning')

    def test_label_wise_partial_fixed_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules that
        predict for a number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-partial-fixed-heads')

    def test_label_wise_partial_fixed_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function and equal-width label binning for
        the induction of rules that predict for a number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-partial-fixed-heads_equal-width-label-binning')

    def test_label_wise_partial_dynamic_heads(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function for the induction of rules that
        predict for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-partial-dynamic-heads')

    def test_label_wise_partial_dynamic_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a label-wise decomposable loss function and equal-width label binning for
        the induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_LABEL_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        self.run_cmd(builder, 'label-wise-partial-dynamic-heads_equal-width-label-binning')

    def test_example_wise_single_label_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules with
        single-label heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_SINGLE_LABEL) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-single-label-heads')

    @SkipTestOnCI
    def test_example_wise_complete_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules with complete
        heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-complete-heads')

    @SkipTestOnCI
    def test_example_wise_complete_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and equal-width label binning for the
        induction of rules with complete heads.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_COMPLETE) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-complete-heads_equal-width-label-binning')

    @SkipTestOnCI
    def test_example_wise_partial_fixed_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules that predict
        for a number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-partial-fixed-heads')

    @SkipTestOnCI
    def test_example_wise_partial_fixed_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and equal-width label binning for the
        induction of rules that predict for a number of labels
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_FIXED) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-partial-fixed-heads_equal-width-label-binning')

    @SkipTestOnCI
    def test_example_wise_partial_dynamic_heads(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function for the induction of rules that predict
        for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_NO) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-partial-dynamic-heads')

    @SkipTestOnCI
    def test_example_wise_partial_dynamic_heads_equal_width_label_binning(self):
        """
        Tests the BOOMER algorithm when using a non-decomposable loss function and equal-width label binning for the
        induction of rules that predict for a dynamically determined subset of the available labels.
        """
        builder = BoostingCmdBuilder() \
            .loss(LOSS_LOGISTIC_EXAMPLE_WISE) \
            .head_type(HEAD_TYPE_PARTIAL_DYNAMIC) \
            .label_binning(LABEL_BINNING_EQUAL_WIDTH) \
            .print_model_characteristics()
        self.run_cmd(builder, 'example-wise-partial-dynamic-heads_equal-width-label-binning')

    def test_global_post_pruning_no_holdout(self):
        """
        Tests the BOOMER algorithm when using no holdout set for global post-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_NO) \
            .print_model_characteristics()
        self.run_cmd(builder, 'post-pruning_no-holdout')

    def test_global_post_pruning_random_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via random sampling for global post-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_RANDOM) \
            .print_model_characteristics()
        self.run_cmd(builder, 'post-pruning_random-holdout')

    def test_global_post_pruning_stratified_label_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via label-wise stratified sampling for
        global post-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_STRATIFIED_LABEL_WISE) \
            .print_model_characteristics()
        self.run_cmd(builder, 'post-pruning_stratified-label-wise-holdout')

    def test_global_post_pruning_stratified_example_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via example-wise stratified sampling for
        global post-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_STRATIFIED_EXAMPLE_WISE) \
            .print_model_characteristics()
        self.run_cmd(builder, 'post-pruning_stratified-example-wise-holdout')

    def test_global_pre_pruning_no_holdout(self):
        """
        Tests the BOOMER algorithm when using no holdout set for global pre-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_NO) \
            .print_model_characteristics()
        self.run_cmd(builder, 'pre-pruning_no-holdout')

    def test_global_pre_pruning_random_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via random sampling for global pre-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_RANDOM) \
            .print_model_characteristics()
        self.run_cmd(builder, 'pre-pruning_random-holdout')

    def test_global_pre_pruning_stratified_label_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via label-wise stratified sampling for
        global pre-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_STRATIFIED_LABEL_WISE) \
            .print_model_characteristics()
        self.run_cmd(builder, 'pre-pruning_stratified-label-wise-holdout')

    def test_global_pre_pruning_stratified_example_wise_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via example-wise stratified sampling for
        global pre-pruning.
        """
        builder = BoostingCmdBuilder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_STRATIFIED_EXAMPLE_WISE) \
            .print_model_characteristics()
        self.run_cmd(builder, 'pre-pruning_stratified-example-wise-holdout')
