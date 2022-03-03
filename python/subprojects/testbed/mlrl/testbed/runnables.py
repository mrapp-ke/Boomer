#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter, DataCharacteristicsLogOutput, \
    DataCharacteristicsCsvOutput
from mlrl.testbed.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from mlrl.testbed.experiments import Experiment
from mlrl.testbed.model_characteristics import RulePrinter, ModelPrinterLogOutput, ModelPrinterTxtOutput, \
    RuleModelCharacteristicsPrinter, RuleModelCharacteristicsLogOutput, RuleModelCharacteristicsCsvOutput
from mlrl.testbed.parameters import ParameterCsvInput
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.training import DataSet

LOG_FORMAT = '%(levelname)s %(message)s'


class Runnable(ABC):
    """
    A base class for all programs that can be configured via command line arguments.
    """

    def run(self, parser: ArgumentParser):
        args = parser.parse_args()

        # Configure the logger...
        log_level = args.log_level
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter(LOG_FORMAT))
        root.addHandler(out_handler)

        log.info('Configuration: %s', args)
        self._run(args)

    @abstractmethod
    def _run(self, args):
        """
        Must be implemented by subclasses in order to run the program.

        :param args: The command line arguments
        """
        pass


class RuleLearnerRunnable(Runnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    def _run(self, args):
        parameter_input = None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)
        evaluation_outputs = []
        data_characteristics_printer_outputs = []
        model_characteristics_printer_outputs = []
        model_printer_outputs = []
        output_dir = args.output_dir

        if args.print_data_characteristics:
            data_characteristics_printer_outputs.append(DataCharacteristicsLogOutput())

        if args.print_evaluation:
            evaluation_outputs.append(EvaluationLogOutput())

        if args.print_model_characteristics:
            model_characteristics_printer_outputs.append(RuleModelCharacteristicsLogOutput())

        if args.print_rules:
            model_printer_outputs.append(ModelPrinterLogOutput())

        if output_dir is not None:
            clear_dir = args.current_fold == -1

            if args.store_data_characteristics:
                data_characteristics_printer_outputs.append(
                    DataCharacteristicsCsvOutput(output_dir=output_dir, clear_dir=clear_dir))
                clear_dir = False

            if args.store_evaluation:
                evaluation_outputs.append(
                    EvaluationCsvOutput(output_dir=output_dir, output_predictions=args.store_predictions,
                                        clear_dir=clear_dir))
                clear_dir = False

            if args.store_model_characteristics:
                model_characteristics_printer_outputs.append(RuleModelCharacteristicsCsvOutput(output_dir=output_dir,
                                                                                               clear_dir=clear_dir))
                clear_dir = False

            if args.store_rules:
                model_printer_outputs.append(ModelPrinterTxtOutput(output_dir=output_dir, clear_dir=clear_dir))

        model_dir = args.model_dir
        persistence = None if model_dir is None else ModelPersistence(model_dir)
        learner = self._create_learner(args)

        data_characteristics_printer = DataCharacteristicsPrinter(data_characteristics_printer_outputs) if len(
            data_characteristics_printer_outputs) > 0 else None
        model_printer = RulePrinter(args.print_options, model_printer_outputs) if len(
            model_printer_outputs) > 0 else None
        model_characteristics_printer = RuleModelCharacteristicsPrinter(model_characteristics_printer_outputs) if len(
            model_characteristics_printer_outputs) > 0 else None
        train_evaluation = ClassificationEvaluation(*evaluation_outputs) if args.evaluate_training_data else None
        test_evaluation = ClassificationEvaluation(*evaluation_outputs)
        data_set = DataSet(data_dir=args.data_dir, data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        experiment = Experiment(learner, test_evaluation=test_evaluation, train_evaluation=train_evaluation,
                                data_set=data_set, num_folds=args.folds, current_fold=args.current_fold,
                                parameter_input=parameter_input, model_printer=model_printer,
                                model_characteristics_printer=model_characteristics_printer,
                                data_characteristics_printer=data_characteristics_printer, persistence=persistence)
        experiment.random_state = args.random_state
        experiment.run()

    @abstractmethod
    def _create_learner(self, args):
        """
        Must be implemented by subclasses in order to create the learner.

        :param args:    The command line arguments
        :return:        The learner that has been created
        """
        pass
