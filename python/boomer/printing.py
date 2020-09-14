#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g. to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod

import numpy as np
from boomer.common.rules import RuleModel, RuleList, Rule, Body, EmptyBody, ConjunctiveBody, Head, FullHead, PartialHead

from boomer.common.learners import Learner
from boomer.data import MetaData
from boomer.io import clear_directory, open_writable_txt_file


class ModelPrinterOutput(ABC):
    """
    An abstract base class for all outputs, textual representations of models may be written to.
    """

    @abstractmethod
    def write_model(self, experiment_name: str, model: str, total_folds: int, fold: int = None):
        """
        Write a textual representation of a model to the output.

        :param experiment_name: The name of the experiment
        :param model:           The textual representation of the model
        :param total_folds:         The total number of folds
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used or if the overall results, averaged over all folds, should be written
        """
        pass


class ModelPrinter(ABC):
    """
    An abstract base class for all classes that allow to print a textual representation of a `MLLearner`'s model.
    """

    def __init__(self, *args: ModelPrinterOutput):
        """
        :param args: The outputs, the textual representations of models should be written to
        """
        self.outputs = args

    def print(self, experiment_name: str, meta_data: MetaData, learner: Learner, current_fold: int, num_folds: int):
        """
        Prints a textual representation of a `MLLearner`'s model.

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
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n%s\n'
        log.info(msg, model)


class ModelPrinterTxtOutput(ModelPrinterOutput):
    """
    Writes the textual representation of a model to a TXT file.
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

    def __init__(self, *args: ModelPrinterOutput):
        super().__init__(*args)

    def _format_model(self, meta_data: MetaData, model) -> str:
        return format_model(meta_data, model)


def format_model(meta_data: MetaData, model: RuleModel, print_feature_names: bool = True,
                 print_label_names: bool = True) -> str:
    """
    Formats a specific rule-based model as a text.

    :param meta_data:           The meta data of the training data set
    :param model:               The model to be formatted
    :param print_feature_names: True, if the names of features should be printed, if available, False, if the indices of
                                features should be printed
    :param print_label_names:   True, if the names of labels should be printed, if available, False, if the indices of
                                labels should be printed
    :return:                    The text
    """
    if isinstance(model, RuleList):
        return __format_rule_list(meta_data, model, print_feature_names, print_label_names)
    else:
        raise ValueError('Model has unknown type: ' + type(model).__name__)


def format_rule(meta_data: MetaData, rule: Rule, print_feature_names: bool = True,
                print_label_names: bool = True) -> str:
    """
    Formats a specific rule as a text.

    :param meta_data:           The meta data of the training data set
    :param rule:                The rule to be formatted
    :param print_feature_names: True, if the names of features should be printed, if available, False, if the indices of
                                features should be printed
    :param print_label_names:   True, if the names of labels should be printed, if available, False, if the indices of
                                labels should be printed
    :return:                    The text
    """
    text = __format_body(meta_data, rule.body, print_feature_names=print_feature_names)
    text += ' -> '
    text += __format_head(meta_data, rule.head, print_label_names=print_label_names)
    return text


def __format_rule_list(meta_data: MetaData, rule_list: RuleList, print_feature_names: bool,
                       print_label_names: bool = True) -> str:
    """
    Formats a specific rule list as a text.

    :param meta_data:           The meta data of the training data set
    :param rule_list:           The rule list to be formatted
    :param print_feature_names: True, if the names of features should be printed, if available, False, if the indices of
                                features should be printed
    :param print_label_names:   True, if the names of labels should be printed, if available, False, if the indices of
                                labels should be printed
    :return:                    The text
    """
    text = ''

    for rule in rule_list.rules:
        if len(text) > 0:
            text += '\n'

        text += format_rule(meta_data, rule, print_feature_names=print_feature_names,
                            print_label_names=print_label_names)

    return text


def __format_body(meta_data: MetaData, body: Body, print_feature_names: bool) -> str:
    """
    Formats the body of a rule as a text.

    :param meta_data:           The meta data of the training data set
    :param body:                The body to be formatted
    :param print_feature_names: True, if the names of features should be printed, if available, False, if the indices of
                                features should be printed
    :return:                    The text
    """
    if isinstance(body, EmptyBody):
        return '{}'
    elif isinstance(body, ConjunctiveBody):
        return '{' + __format_conjunctive_body(meta_data, body, print_feature_names=print_feature_names) + '}'
    else:
        raise ValueError('Body has unknown type: ' + type(body).__name__)


def __format_conjunctive_body(meta_data: MetaData, body: ConjunctiveBody, print_feature_names: bool) -> str:
    """
    Formats the conjunctive body of a rule as a text.

    :param meta_data:           The meta data of the training data set
    :param body:                The conjunctive body to be formatted
    :param print_feature_names: True, if the names of features should be printed, if available, False, if the indices of
                                features should be printed
    :return:                    The text
    """
    text = ''

    if body.leq_feature_indices is not None and body.leq_thresholds is not None:
        text = __format_conditions(meta_data, np.asarray(body.leq_feature_indices), np.asarray(body.leq_thresholds),
                                   '<=', text, print_feature_names=print_feature_names)

    if body.gr_feature_indices is not None and body.gr_thresholds is not None:
        text = __format_conditions(meta_data, np.asarray(body.gr_feature_indices), np.asarray(body.gr_thresholds),
                                   '>', text, print_feature_names=print_feature_names)

    if body.eq_feature_indices is not None and body.eq_thresholds is not None:
        text = __format_conditions(meta_data, np.asarray(body.eq_feature_indices), np.asarray(body.eq_thresholds),
                                   '==', text, print_feature_names=print_feature_names)

    if body.neq_feature_indices is not None and body.neq_thresholds is not None:
        text = __format_conditions(meta_data, np.asarray(body.neq_feature_indices), np.asarray(body.neq_thresholds),
                                   '!=', text, print_feature_names=print_feature_names)

    return text


def __format_conditions(meta_data: MetaData, feature_indices: np.ndarray, thresholds: np.ndarray, operator: str,
                        text: str, print_feature_names: bool) -> str:
    """
    Formats conditions that are contained by the body of a rule and the textual representation to an existing text.

    :param meta_data:           The meta data of the training data set
    :param feature_indices:     An array of type `int`, shape `(num_conditions)`, representing the feature indices that
                                correspond to the conditions
    :param thresholds:          An array of type `float`, shape `(num_conditions)`, representing the thresholds used by
                                the conditions
    :param operator:            A textual representation of the operator that is used by the conditions
    :param text:                The text, the textual representation of the conditions should be appended to
    :param print_feature_names: True, if the names of features should be printed, if available, False, if the indices of
                                features should be printed
    :return:                    The given text including the appended text
    """
    for i in range(feature_indices.shape[0]):
        if len(text) > 0:
            text += ' & '

        feature_index = feature_indices[i]

        if print_feature_names and len(meta_data.attributes) > feature_index:
            text += meta_data.attributes[feature_index].attribute_name
        else:
            text += str(feature_index)

        text += ' '
        text += operator
        text += ' '
        text += str(thresholds[i])

    return text


def __format_head(meta_data: MetaData, head: Head, print_label_names: bool) -> str:
    """
    Formats the head of a rule as a text.

    :param meta_data:           The meta data of the training data set
    :param head:                The head to be formatted
    :param print_label_names:   True, if the names of labels should be printed, if available, False, if the indices of
                                labels should be printed
    :return:                    The text
    """
    if isinstance(head, FullHead):
        return '(' + __format_full_head(meta_data, head, print_label_names=print_label_names) + ')'
    elif isinstance(head, PartialHead):
        return '(' + __format_partial_head(meta_data, head, print_label_names=print_label_names) + ')'
    else:
        raise ValueError('Head has unknown type: ' + type(head).__name__)


def __format_full_head(meta_data: MetaData, head: FullHead, print_label_names: bool) -> str:
    """
    Formats the full head of a rule as a text.

    :param meta_data:           The meta data of the training data set
    :param head:                The full head to be formatted
    :param print_label_names:   True, if the names of labels should be printed, if available, False, if the indices of
                                labels should be printed
    :return:                    The text
    """
    text = ''
    scores = np.asarray(head.scores)

    for i in range(scores.shape[0]):
        if len(text) > 0:
            text += ', '

        if print_label_names and len(meta_data.label_names) > i:
            text += meta_data.label_names[i]
        else:
            text += str(i)

        text += ' = '
        text += '{0:.2f}'.format(scores[i])

    return text


def __format_partial_head(meta_data: MetaData, head: PartialHead, print_label_names: bool) -> str:
    """
    Formats the partial head of a rule as a text.

    :param meta_data:           The meta data of the training data set
    :param head:                The partial head to be formatted
    :param print_label_names:   True, if the names of labels should be printed, if available, False, if the indices of
                                labels should be printed
    :return:                    The text
    """
    text = ''
    scores = np.asarray(head.scores)
    label_indices = np.asarray(head.label_indices)

    for i in range(label_indices.shape[0]):
        if len(text) > 0:
            text += ', '

        label_index = label_indices[i]

        if print_label_names and len(meta_data.label_names) > label_index:
            text += meta_data.label_names[label_index]
        else:
            text += str(label_index)

        text += ' = '
        text += '{0:.2f}'.format(scores[i])

    return text
