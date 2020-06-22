#!/usr/bin/python

import argparse
import logging as log
import os.path as path

import numpy as np
from skmultilearn.base import MLClassifierBase

from boomer.algorithm.model import Theory, DTYPE_FLOAT32, DTYPE_FLOAT64
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import HAMMING_LOSS, SUBSET_01_LOSS
from boomer.persistence import ModelPersistence
from boomer.plots import LossMinimizationCurve
from boomer.training import CrossValidation
from main_boomer import configure_argument_parser, create_learner


class Plotter(CrossValidation, MLClassifierBase):
    """
    Plots the performance of a BOOMER model at each iteration.
    """

    def __init__(self, model_dir: str, output_dir: str, data_dir: str, data_set: str, num_folds: int, current_fold: int,
                 learner: Boomer):
        super().__init__(data_dir, data_set, num_folds, current_fold)
        self.output_dir = output_dir
        self.require_dense = [True, True]  # We need a dense representation of the training data
        self.persistence = ModelPersistence(model_dir=model_dir)
        self.learner = learner
        self.plot = LossMinimizationCurve(data_set, HAMMING_LOSS, SUBSET_01_LOSS)

    def _train_and_evaluate(self, train_indices, train_x, train_y, test_indices, test_x, test_y, first_fold: int,
                            current_fold: int, last_fold: int, num_folds: int):
        # Create a dense representation of the training data
        train_x = np.asfortranarray(self._ensure_input_format(train_x), dtype=DTYPE_FLOAT32)
        train_y = self._ensure_input_format(train_y)
        test_x = np.asfortranarray(self._ensure_input_format(test_x), dtype=DTYPE_FLOAT32)
        test_y = self._ensure_input_format(test_y)

        learner = self.learner
        theory: Theory = self.persistence.load_model(model_name=learner.get_model_name(),
                                                     file_name_suffix=learner.get_model_prefix(),
                                                     fold=current_fold, raise_exception=True)
        num_iterations = len(theory)

        train_predictions = np.asfortranarray(np.zeros((train_x.shape[0], train_y.shape[1]), dtype=DTYPE_FLOAT64))
        test_predictions = np.asfortranarray(np.zeros((test_x.shape[0], test_y.shape[1]), dtype=DTYPE_FLOAT64))

        for i in range(1, num_iterations + 1):
            log.info("Evaluating model at iteration %s / %s...", i, num_iterations)
            rule = theory.pop(0)

            rule.predict(train_x, train_predictions)
            self.plot.add_evaluation(np.where(train_predictions > 0, 1, 0), train_y, training_data=True, num_rules=i,
                                     first_fold=first_fold, current_fold=current_fold, last_fold=last_fold,
                                     num_folds=num_folds)

            rule.predict(test_x, test_predictions)
            self.plot.add_evaluation(np.where(test_predictions > 0, 1, 0), test_y, training_data=False, num_rules=i,
                                     first_fold=first_fold, current_fold=current_fold, last_fold=last_fold,
                                     num_folds=num_folds)

        if current_fold == last_fold:
            self.__plot()

    def __plot(self):
        output_dir = self.output_dir

        if output_dir is not None:
            file_name = self.learner.get_name() + '.pdf'
            output_file = path.join(self.output_dir, file_name)
        else:
            output_file = None

        self.plot.plot(output_file)

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    @staticmethod
    def __get_experiment_name(prefix: str, iteration: int) -> str:
        name = prefix + ('_' if len(prefix) > 0 else '')
        name += 'iteration-' + str(iteration)
        return name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots the performance of a BOOMER model')
    configure_argument_parser(parser)
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    plotter = Plotter(model_dir=args.model_dir, output_dir=args.output_dir, data_dir=args.data_dir,
                      data_set=args.dataset, num_folds=args.folds, current_fold=args.current_fold,
                      learner=create_learner(args))
    plotter.run()
