#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g. to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import List, Set

from mlrl.common.cython.model import RuleModelFormatter

from mlrl.common.learners import Learner
from mlrl.common.options import Options
from mlrl.testbed.data import MetaData
from mlrl.testbed.io import clear_directory, open_writable_txt_file

ARGUMENT_PRINT_FEATURE_NAMES = 'print_feature_names'

ARGUMENT_PRINT_LABEL_NAMES = 'print_label_names'

ARGUMENT_PRINT_NOMINAL_VALUES = 'print_nominal_values'

PRINT_OPTION_VALUES: Set[str] = {ARGUMENT_PRINT_FEATURE_NAMES, ARGUMENT_PRINT_LABEL_NAMES,
                                 ARGUMENT_PRINT_NOMINAL_VALUES}


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
    An abstract base class for all classes that allow to print a textual representation of a `MLLearner`'s model.
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
        formatter.format(model)
        return formatter.get_text()
