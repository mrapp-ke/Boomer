#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g. to the console or to a file.
"""
import logging as log
from _io import StringIO
from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np
from mlrl.common.cython.rule_model import RuleModelVisitor, EmptyBody, ConjunctiveBody, CompleteHead, PartialHead
from mlrl.common.learners import Learner
from mlrl.common.options import Options
from mlrl.testbed.data import Attribute, MetaData
from mlrl.testbed.io import clear_directory, open_writable_txt_file, open_writable_csv_file, create_csv_dict_writer

ARGUMENT_PRINT_FEATURE_NAMES = 'print_feature_names'

ARGUMENT_PRINT_LABEL_NAMES = 'print_label_names'

ARGUMENT_PRINT_NOMINAL_VALUES = 'print_nominal_values'

PRINT_OPTION_VALUES: Set[str] = {ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES,
                                 ARGUMENT_PRINT_NOMINAL_VALUES}


class RuleModelFormatter(RuleModelVisitor):
    """
    Allows to create textual representation of the rules in a `RuleModel`.
    """

    def __init__(self, attributes: List[Attribute], labels: List[Attribute], print_feature_names: bool,
                 print_label_names: bool, print_nominal_values: bool):
        """
        :param attributes:              A list that contains the attributes
        :param labels:                  A list that contains the labels
        :param print_feature_names:     True, if the names of features should be printed, False otherwise
        :param print_label_names:       True, if the names of labels should be printed, False otherwise
        :param print_nominal_values:    True, if the values of nominal values should be printed, False otherwise
        """
        self.print_feature_names = print_feature_names
        self.print_label_names = print_label_names
        self.print_nominal_values = print_nominal_values
        self.attributes = attributes
        self.labels = labels
        self.text = StringIO()

    def visit_empty_body(self, _: EmptyBody):
        self.text.write('{}')

    def __format_conditions(self, num_conditions: int, indices: np.ndarray, thresholds: np.ndarray,
                            operator: str) -> int:
        result = num_conditions

        if indices is not None and thresholds is not None:
            text = self.text
            attributes = self.attributes
            print_feature_names = self.print_feature_names
            print_nominal_values = self.print_nominal_values

            for i in range(indices.shape[0]):
                if result > 0:
                    text.write(' & ')

                feature_index = indices[i]
                threshold = thresholds[i]
                attribute = attributes[feature_index] if len(attributes) > feature_index else None

                if print_feature_names and attribute is not None:
                    text.write(attribute.attribute_name)
                else:
                    text.write(str(feature_index))

                text.write(' ')
                text.write(operator)
                text.write(' ')

                if attribute is not None and attribute.nominal_values is not None:
                    if print_nominal_values and len(attribute.nominal_values) > threshold:
                        text.write('"' + attribute.nominal_values[threshold] + '"')
                    else:
                        text.write(str(threshold))
                else:
                    text.write(str(threshold))

                result += 1

        return result

    def visit_conjunctive_body(self, body: ConjunctiveBody):
        text = self.text
        text.write('{')
        num_conditions = self.__format_conditions(0, body.leq_indices, body.leq_thresholds, '<=')
        num_conditions = self.__format_conditions(num_conditions, body.gr_indices, body.gr_thresholds, '>')
        num_conditions = self.__format_conditions(num_conditions, body.eq_indices, body.eq_thresholds, '==')
        self.__format_conditions(num_conditions, body.neq_indices, body.neq_thresholds, '!=')
        text.write('}')

    def visit_complete_head(self, head: CompleteHead):
        text = self.text
        print_label_names = self.print_label_names
        labels = self.labels
        scores = head.scores
        text.write(' => (')

        for i in range(scores.shape[0]):
            if i > 0:
                text.write(', ')

            if print_label_names and len(labels) > i:
                text.write(labels[i].attribute_name)
            else:
                text.write(str(i))

            text.write(' = ')
            text.write('{0:.2f}'.format(scores[i]))

        text.write(')\n')

    def visit_partial_head(self, head: PartialHead):
        text = self.text
        print_label_names = self.print_label_names
        labels = self.labels
        indices = head.indices
        scores = head.scores
        text.write(' => (')

        for i in range(indices.shape[0]):
            if i > 0:
                text.write(', ')

            label_index = indices[i]

            if print_label_names and len(labels) > label_index:
                text.write(labels[label_index].attribute_name)
            else:
                text.write(str(label_index))

            text.write(' = ')
            text.write('{0:.2f}'.format(scores[i]))

        text.write(')\n')

    def get_text(self) -> str:
        """
        Returns the textual representation that has been created via the `format` method.

        :return: The textual representation
        """
        return self.text.getvalue()


class ModelPrinterOutput(ABC):
    """
    An abstract base class for all outputs, textual representations of models may be written to.
    """

    @abstractmethod
    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        """
        Write a textual representation of a model to the output.

        :param experiment_name:     The name of the experiment
        :param model:               The textual representation of the model
        :param total_folds:         The total number of folds
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used or if the overall results, averaged over all folds, should be written
        """
        pass


class ModelPrinter(ABC):
    """
    An abstract base class for all classes that allow to print a textual representation of a `Learner`'s model.
    """

    def __init__(self, print_options: str, outputs: List[ModelPrinterOutput]):
        """
        :param print_options:   The options to be used for printing models
        :param outputs:         The outputs, the textual representations of models should be written to
        """
        self.outputs = outputs

        try:
            self.print_options = Options.create(print_options, PRINT_OPTION_VALUES)
        except ValueError as e:
            raise ValueError('Invalid value given for parameter "print_options". ' + str(e))

    def print(self, experiment_name: str, meta_data: MetaData, learner: Learner, current_fold: int, num_folds: int):
        """
        Prints a textual representation of a `Learner`'s model.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the training data set
        :param learner:         The learner
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        model = learner.model_
        text = self._format_model(meta_data, model)

        for output in self.outputs:
            output.write_model(experiment_name, text, num_folds, current_fold if num_folds > 1 else None)

    @abstractmethod
    def _format_model(self, meta_data: MetaData, model) -> str:
        """
        Must be implemented by subclasses in order to create a textual representation of a model.

        :param meta_data:   The meta data of the training data set
        :param model:       The model
        :return:            The textual representation of the given model
        """
        pass


class ModelPrinterLogOutput(ModelPrinterOutput):
    """
    Outputs the textual representation of a model using the logger.
    """

    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        msg = 'Model for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n%s'
        log.info(msg, model)


class ModelPrinterTxtOutput(ModelPrinterOutput):
    """
    Writes the textual representation of a model to a text file.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True):
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        with open_writable_txt_file(self.output_dir, 'rules', fold, append=False) as text_file:
            text_file.write(model)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class RulePrinter(ModelPrinter):
    """
    Allows to print a textual representation of a `MLRuleLearner`'s rule-based model.
    """

    def __init__(self, print_options: str, outputs: List[ModelPrinterOutput]):
        super().__init__(print_options, outputs)

    def _format_model(self, meta_data: MetaData, model) -> str:
        print_options = self.print_options
        print_feature_names = print_options.get_bool(ARGUMENT_PRINT_FEATURE_NAMES, True)
        print_label_names = print_options.get_bool(ARGUMENT_PRINT_LABEL_NAMES, True)
        print_nominal_values = print_options.get_bool(ARGUMENT_PRINT_NOMINAL_VALUES, True)
        formatter = RuleModelFormatter(attributes=meta_data.attributes, labels=meta_data.labels,
                                       print_feature_names=print_feature_names, print_label_names=print_label_names,
                                       print_nominal_values=print_nominal_values)
        model.visit(formatter)
        return formatter.get_text()


class RuleModelCharacteristics:
    """
    Stores the characteristics of a `RuleModel`.
    """

    def __init__(self, default_rule_index: int, default_rule_pos_predictions: int, default_rule_neg_predictions: int,
                 num_leq: np.ndarray, num_gr: np.ndarray, num_eq: np.ndarray, num_neq: np.ndarray,
                 num_pos_predictions: np.ndarray, num_neg_predictions: np.ndarray):
        """
        :param default_rule_index:              The index of the default rule or None, if no default rule is used
        :param default_rule_pos_predictions:    The number of positive predictions of the default rule, if any
        :param default_rule_neg_predictions:    The number of negative predictions of the default rule, if any
        :param num_leq:                         A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the <= operator per rule
        :param num_gr:                          A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the > operator per rule
        :param num_eq:                          A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the == operator per rule
        :param num_neq:                         A `np.ndarray`, shape `(num_rules)` that stores the number of conditions
                                                that use the != operator per rule
        :param num_pos_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of positive
                                                predictions per rule
        :param num_neg_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of negative
                                                predictions per rule
        """
        self.default_rule_index = default_rule_index
        self.default_rule_pos_predictions = default_rule_pos_predictions
        self.default_rule_neg_predictions = default_rule_neg_predictions
        self.num_leq = num_leq
        self.num_gr = num_gr
        self.num_eq = num_eq
        self.num_neq = num_neq
        self.num_pos_predictions = num_pos_predictions
        self.num_neg_predictions = num_neg_predictions


class RuleModelCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of a `MLRuleLearner`'s model may be written to.
    """

    @abstractmethod
    def write_model_characteristics(self, experiment_name: str, characteristics: RuleModelCharacteristics,
                                    total_folds: int, fold: int = None):
        """
        Writes the characteristics of a `RuleModel` to the output.

        :param experiment_name: The name of the experiment
        :param characteristics: The characteristics of the model
        :param total_folds:     The total number of folds
        :param fold:            The fold for which the characteristics should be written or None, if no cross validation
                                is used
        """
        pass


class RuleModelCharacteristicsVisitor(RuleModelVisitor):
    """
    A visitor that allows to determine the characteristics of a `RuleModel`.
    """

    def __init__(self):
        self.num_leq = []
        self.num_gr = []
        self.num_eq = []
        self.num_neq = []
        self.num_pos_predictions = []
        self.num_neg_predictions = []
        self.default_rule_index = None
        self.default_rule_pos_predictions = 0
        self.default_rule_neg_predictions = 0
        self.index = -1

    def visit_empty_body(self, _: EmptyBody):
        self.index += 1
        self.default_rule_index = self.index

    def visit_conjunctive_body(self, body: ConjunctiveBody):
        self.index += 1
        self.num_leq.append(body.leq_indices.shape[0] if body.leq_indices is not None else 0)
        self.num_gr.append(body.gr_indices.shape[0] if body.gr_indices is not None else 0)
        self.num_eq.append(body.eq_indices.shape[0] if body.eq_indices is not None else 0)
        self.num_neq.append(body.neq_indices.shape[0] if body.neq_indices is not None else 0)

    def visit_complete_head(self, head: CompleteHead):
        num_pos_predictions = np.count_nonzero(head.scores > 0)
        num_neg_predictions = head.scores.shape[0] - num_pos_predictions

        if self.index == self.default_rule_index:
            self.default_rule_pos_predictions = num_pos_predictions
            self.default_rule_neg_predictions = num_neg_predictions
        else:
            self.num_pos_predictions.append(num_pos_predictions)
            self.num_neg_predictions.append(num_neg_predictions)

    def visit_partial_head(self, head: PartialHead):
        num_pos_predictions = np.count_nonzero(head.scores > 0)
        num_neg_predictions = head.scores.shape[0] - num_pos_predictions

        if self.index == self.default_rule_index:
            self.default_rule_pos_predictions = num_pos_predictions
            self.default_rule_neg_predictions = num_neg_predictions
        else:
            self.num_pos_predictions.append(num_pos_predictions)
            self.num_neg_predictions.append(num_neg_predictions)


class RuleModelCharacteristicsLogOutput(RuleModelCharacteristicsOutput):
    """
    Outputs the characteristics of a `RuleModel` using the logger.
    """

    def write_model_characteristics(self, experiment_name: str, characteristics: RuleModelCharacteristics,
                                    total_folds: int, fold: int = None):
        default_rule_index = characteristics.default_rule_index
        num_pos_predictions = characteristics.num_pos_predictions
        num_neg_predictions = characteristics.num_neg_predictions
        num_predictions = num_pos_predictions + num_neg_predictions
        num_leq = characteristics.num_leq
        num_gr = characteristics.num_gr
        num_eq = characteristics.num_eq
        num_neq = characteristics.num_neq
        num_conditions = num_leq + num_gr + num_eq + num_neq
        num_total_conditions = np.sum(num_conditions)
        frac_leq = np.sum(num_leq) / num_total_conditions * 100
        frac_gr = np.sum(num_gr) / num_total_conditions * 100
        frac_eq = np.sum(num_eq) / num_total_conditions * 100
        frac_neq = 100 - frac_leq - frac_gr - frac_eq
        num_total_predictions = np.sum(num_predictions)
        frac_pos = np.sum(num_pos_predictions) / num_total_predictions * 100
        frac_neg = 100 - frac_pos
        num_rules = num_predictions.shape[0]
        msg = 'Model characteristics for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n'
        msg += 'Rules: ' + str(num_rules)
        if default_rule_index is not None:
            msg += ' (plus a default rule with ' + str(
                characteristics.default_rule_pos_predictions) + ' positive and ' + str(
                characteristics.default_rule_neg_predictions) + ' negative predictions that is excluded from the ' \
                   + 'following statistics)'
        msg += '\n'
        msg += 'Conditions per rule: avg. ' + str(np.mean(num_conditions)) + ', min. ' + str(
            np.min(num_conditions)) + ', max. ' + str(np.max(num_conditions)) + '\n'
        msg += 'Conditions total: ' + str(num_total_conditions) + ' (' + str(frac_leq) + '% use <= operator, ' + str(
            frac_gr) + '% use > operator, ' + str(frac_eq) + '% use == operator, ' + str(
            frac_neq) + '% use != operator)\n'
        msg += 'Predictions per rule: avg. ' + str(np.mean(num_predictions)) + ', min. ' + str(
            np.min(num_predictions)) + ', max. ' + str(np.max(num_predictions)) + '\n'
        msg += 'Predictions total: ' + str(num_total_predictions) + ' (' + str(frac_pos) + '% positive, ' + str(
            frac_neg) + '% negative)\n'
        log.info(msg)


class RuleModelCharacteristicsCsvOutput(RuleModelCharacteristicsOutput):
    """
    Writes the characteristics of a `RuleModel` to a CSV file.
    """

    COL_RULE_NAME = 'Rule'

    COL_CONDITIONS = 'conditions'

    COL_NUMERICAL_CONDITIONS = 'numerical conditions'

    COL_LEQ_CONDITIONS = 'conditions using <= operator'

    COL_GR_CONDITIONS = 'conditions using > operator'

    COL_NOMINAL_CONDITIONS = 'nominal conditions'

    COL_EQ_CONDITIONS = 'conditions using == operator'

    COL_NEQ_CONDITIONS = 'conditions using != operator'

    COL_PREDICTIONS = 'predictions'

    COL_POS_PREDICTIONS = 'pos. predictions'

    COL_NEG_PREDICTIONS = 'neg. predictions'

    def __init__(self, output_dir: str, clear_dir: bool = True):
        """
        :param output_dir:  The path of the directory, the CSV files should be written to
        :param clear_dir:   True, if the directory, the CSV files should be written to, should be cleared
        """
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_model_characteristics(self, experiment_name: str, characteristics: RuleModelCharacteristics,
                                    total_folds: int, fold: int = None):
        if fold is not None:
            self.__clear_dir_if_necessary()
            header = [
                RuleModelCharacteristicsCsvOutput.COL_RULE_NAME,
                RuleModelCharacteristicsCsvOutput.COL_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_NUMERICAL_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_LEQ_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_GR_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_NOMINAL_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_EQ_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_NEQ_CONDITIONS,
                RuleModelCharacteristicsCsvOutput.COL_PREDICTIONS,
                RuleModelCharacteristicsCsvOutput.COL_POS_PREDICTIONS,
                RuleModelCharacteristicsCsvOutput.COL_NEG_PREDICTIONS
            ]
            default_rule_index = characteristics.default_rule_index
            num_rules = len(characteristics.num_pos_predictions)
            num_total_rules = num_rules if default_rule_index is None else num_rules + 1

            with open_writable_csv_file(self.output_dir, 'model_characteristics', fold) as csv_file:
                csv_writer = create_csv_dict_writer(csv_file, header)
                n = 0

                for i in range(num_total_rules):
                    rule_name = 'Rule ' + str(i + 1)

                    if i == default_rule_index:
                        rule_name += ' (Default rule)'
                        num_leq = 0
                        num_gr = 0
                        num_eq = 0
                        num_neq = 0
                        num_pos_predictions = characteristics.default_rule_pos_predictions
                        num_neg_predictions = characteristics.default_rule_neg_predictions
                    else:
                        num_leq = characteristics.num_leq[n]
                        num_gr = characteristics.num_gr[n]
                        num_eq = characteristics.num_eq[n]
                        num_neq = characteristics.num_neq[n]
                        num_pos_predictions = characteristics.num_pos_predictions[n]
                        num_neg_predictions = characteristics.num_neg_predictions[n]
                        n += 1

                    num_numerical = num_leq + num_gr
                    num_nominal = num_eq + num_neq
                    num_conditions = num_numerical + num_nominal
                    num_predictions = num_pos_predictions + num_neg_predictions
                    columns = {
                        RuleModelCharacteristicsCsvOutput.COL_RULE_NAME: rule_name,
                        RuleModelCharacteristicsCsvOutput.COL_CONDITIONS: num_conditions,
                        RuleModelCharacteristicsCsvOutput.COL_NUMERICAL_CONDITIONS: num_numerical,
                        RuleModelCharacteristicsCsvOutput.COL_LEQ_CONDITIONS: num_leq,
                        RuleModelCharacteristicsCsvOutput.COL_GR_CONDITIONS: num_gr,
                        RuleModelCharacteristicsCsvOutput.COL_NOMINAL_CONDITIONS: num_nominal,
                        RuleModelCharacteristicsCsvOutput.COL_EQ_CONDITIONS: num_eq,
                        RuleModelCharacteristicsCsvOutput.COL_NEQ_CONDITIONS: num_neq,
                        RuleModelCharacteristicsCsvOutput.COL_PREDICTIONS: num_predictions,
                        RuleModelCharacteristicsCsvOutput.COL_POS_PREDICTIONS: num_pos_predictions,
                        RuleModelCharacteristicsCsvOutput.COL_NEG_PREDICTIONS: num_neg_predictions
                    }
                    csv_writer.writerow(columns)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class ModelCharacteristicsPrinter(ABC):
    """
    A class that allows to print the characteristics of a `Learner`'s model.
    """

    def print(self, experiment_name: str, learner: Learner, current_fold: int, num_folds: int):
        model = learner.model_
        self._print_model(experiment_name, current_fold, num_folds, model)

    def _print_model(self, experiment_name: str, current_fold: int, num_folds: int, model):
        """
        :param experiment_name: The name of the experiment
        :param current_fold:    The current fold
        :param num_folds:       The total number of folds
        :param model:           The model
        """
        pass


class RuleModelCharacteristicsPrinter(ModelCharacteristicsPrinter):
    """
    A class that allows to print the characteristics of `MLRuleLearner`'s model.
    """

    def __init__(self, outputs: List[RuleModelCharacteristicsOutput]):
        """
        :param outputs: The outputs, the characteristics of `RuleModel`s should be written to
        """
        self.outputs = outputs

    def _print_model(self, experiment_name: str, current_fold: int, num_folds: int, model):
        if len(self.outputs) > 0:
            visitor = RuleModelCharacteristicsVisitor()
            model.visit(visitor)
            characteristics = RuleModelCharacteristics(
                default_rule_index=visitor.default_rule_index,
                default_rule_pos_predictions=visitor.default_rule_pos_predictions,
                default_rule_neg_predictions=visitor.default_rule_neg_predictions, num_leq=np.asarray(visitor.num_leq),
                num_gr=np.asarray(visitor.num_gr), num_eq=np.asarray(visitor.num_eq),
                num_neq=np.asarray(visitor.num_neq), num_pos_predictions=np.asarray(visitor.num_pos_predictions),
                num_neg_predictions=np.asarray(visitor.num_neg_predictions))

            for output in self.outputs:
                output.write_model_characteristics(experiment_name, characteristics, num_folds,
                                                   current_fold if num_folds > 1 else None)
