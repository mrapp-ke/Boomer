#!/usr/bin/python

"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for creating plots.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from boomer.evaluation import ClassificationEvaluation


class Plot(ABC):
    """
    A base class for all plots.
    """

    def plot(self, output_file: str = None):
        """
        Creates the plot and writes it to a specific output file, or displays it in a window.

        :param output_file: The path of the output file or None, if the plot should be displayed in a window
        """
        log.debug('Creating plot...')
        self._prepare_plot()
        log.info('Successfully created plot.')

        if output_file is not None:
            log.debug('Saving plot to file \'' + output_file + '\'...')
            plt.savefig(output_file)
            log.info('Successfully saved plot to file \'' + output_file + '\'.')
        else:
            log.info('Displaying plot in a window...')
            plt.show()

    @abstractmethod
    def _prepare_plot(self):
        """
        Must be implemented by subclasses in order to prepare the plot.
        """
        pass


class EvaluationPlot(Plot, ABC):
    """
    A base class for all plots that visualize the performance of classifiers according to different evaluation measures.
    """

    def __init__(self, *args: str):
        """
        :param args: The names of the evaluation measures to be used
        """
        self.measures = args
        self.evaluations: Dict[str, ClassificationEvaluation] = {}

    def _add_evaluation(self, predictions, ground_truth, training_data: bool, classifier_name: str, first_fold: int,
                        current_fold: int, last_fold: int, num_folds: int):
        """
        Updates the plot by evaluating the quality of predictions that have been provided by a certain classifier.
        the plot.

        :param predictions:     The predictions provided by the classifier
        :param ground_truth:    The true labels
        :param training_data:   True, if the predictions have been made for the training data, False, if the have been
                                made for the test data
        :param classifier_name: The name of the classifier that provided the predictions
        :param first_fold:      The first cross validation fold or 0, if no cross validation is used
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param last_fold:       The last cross validation fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        key = 'train' if training_data else 'test'
        evaluations = self.evaluations
        evaluation = evaluations[key] if key in evaluations else ClassificationEvaluation()
        evaluation.evaluate(classifier_name, predictions, ground_truth, first_fold=first_fold,
                            current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)
        evaluations[key] = evaluation


class NumRulesDependentEvaluationPlot(EvaluationPlot, ABC):
    """
    A base class for all plots that visualize the performance of a classifier depending on the number of rules in the
    model.
    """

    def add_evaluation(self, predictions, ground_truth, training_data: bool, num_rules: int, first_fold: int,
                       current_fold: int, last_fold: int, num_folds: int):
        """
        Updates the plot by evaluating the quality of predictions that have been provided by a certain classifier.
        the plot.

        :param predictions:     The predictions provided by the classifier
        :param ground_truth:    The true labels
        :param training_data:   True, if the predictions have been made for the training data, False, if the have been
                                made for the test data
        :param num_rules:       The number of rules contained in the model
        :param first_fold:      The first cross validation fold or 0, if no cross validation is used
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param last_fold:       The last cross validation fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        self._add_evaluation(predictions, ground_truth, training_data=training_data, classifier_name=str(num_rules),
                             first_fold=first_fold, current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)


class LossMinimizationCurve(NumRulesDependentEvaluationPlot):
    """
    A plot that uses curves to visualize how certain losses are minimized as more rules are added to a model, i.e., the
    y-axis corresponds to the evaluation score and the x-axis corresponds to the number of rules in the model.
    """

    def __init__(self, data_set: str, *args: str):
        super().__init__(*args)
        self.data_set = data_set

    def __plot_curves(self):
        max_num_rules = 0

        # Draw curves
        for key, evaluation in self.evaluations.items():
            evaluation_names = sorted([int(x) for x in evaluation.results.keys()])

            for measure in self.measures:
                x = []
                y = []

                for num_rules in evaluation_names:
                    max_num_rules = max(max_num_rules, num_rules)
                    evaluation_result = evaluation.results[str(num_rules)]
                    score, std_dev = evaluation_result.avg(measure)
                    x.append(num_rules)
                    y.append(score)

                plt.plot(x, y, label=(measure + ' (' + key + ')'))

        plt.legend()
        return max_num_rules

    def _prepare_plot(self):
        max_x = self.__plot_curves()

        # Set title
        plt.title(self.data_set)

        # Customize x axis
        plt.xlabel('# rules')
        plt.xlim(left=0, right=max_x)
        x_ticks = np.arange(0, max_x + 200, 200)
        plt.xticks(ticks=x_ticks)

        # Draw vertical lines
        prev_x = None

        for x in x_ticks:
            if prev_x is not None:
                new_x = prev_x + ((x - prev_x) / 2)
                plt.plot([new_x, new_x], [0, 1.0], color='0.5', linestyle='dotted', linewidth=1)
            if 0 < x < max_x:
                plt.plot([x, x], [0, 1.0], color='0.5', linestyle='dotted', linewidth=1)
            prev_x = x

        # Customize y axis
        plt.ylim(bottom=0, top=1)
        y_labels = [str(i * 10) + '%' for i in range(11)]
        y_ticks = np.arange(0, 1.1, 0.1)
        plt.yticks(ticks=y_ticks, labels=y_labels)

        # Draw horizontal lines
        prev_y = None

        for y in y_ticks:
            if prev_y is not None:
                new_y = prev_y + ((y - prev_y) / 2)
                plt.plot([0, max_x], [new_y, new_y], color='0.5', linestyle='dotted', linewidth=1)
            if 0 < y < 1.0:
                plt.plot([0, max_x], [y, y], color='0.5', linestyle='dotted', linewidth=1)
            prev_y = y


class MeasureVsMeasureCurve(NumRulesDependentEvaluationPlot):
    """
    A plot that uses a single curve to visualize how the trade-off between two evaluation measures is affected as more
    rules are added to a model, i.e., the y-axis corresponds to one of the measures and the x-axis corresponds to the
    other one.
    """

    def __init__(self, data_set: str, first_measure: str, second_measure: str):
        super().__init__(first_measure, second_measure)
        self.data_set = data_set

    def __plot_curves(self) -> (str, str):
        y_measure = self.measures[0]
        x_measure = self.measures[1]

        # Draw curves
        for key, evaluation in self.evaluations.items():
            evaluation_names = sorted([int(x) for x in evaluation.results.keys()])
            x = []
            y = []

            for num_rules in evaluation_names:
                evaluation_result = evaluation.results[str(num_rules)]
                score_x, std_dev_x = evaluation_result.avg(x_measure)
                x.append(score_x)
                score_y, std_dev_y = evaluation_result.avg(y_measure)
                y.append(score_y)

            plt.plot(x, y, label=key)

        plt.legend()
        return x_measure, y_measure

    def _prepare_plot(self):
        x_measure, y_measure = self.__plot_curves()

        # Set title
        plt.title(self.data_set)

        # Customize x axis
        plt.xlabel(x_measure)
        plt.xlim(left=0, right=1)
        x_labels = [str(i * 10) + '%' for i in range(11)]
        x_ticks = np.arange(0, 1.1, 0.1)
        plt.xticks(ticks=x_ticks, labels=x_labels)

        # Draw vertical lines
        prev_x = None

        for x in x_ticks:
            if prev_x is not None:
                new_x = prev_x + ((x - prev_x) / 2)
                plt.plot([new_x, new_x], [0, 1.0], color='0.5', linestyle='dotted', linewidth=1)
            if 0 < x < 1:
                plt.plot([x, x], [0, 1.0], color='0.5', linestyle='dotted', linewidth=1)
            prev_x = x

        # Customize y axis
        plt.ylabel(y_measure)
        plt.ylim(bottom=0, top=1)
        y_labels = [str(i * 10) + '%' for i in range(11)]
        y_ticks = np.arange(0, 1.1, 0.1)
        plt.yticks(ticks=y_ticks, labels=y_labels)

        # Draw horizontal lines
        prev_y = None

        for y in y_ticks:
            if prev_y is not None:
                new_y = prev_y + ((y - prev_y) / 2)
                plt.plot([0, 1], [new_y, new_y], color='0.5', linestyle='dotted', linewidth=1)
            if 0 < y < 1.0:
                plt.plot([0, 1], [y, y], color='0.5', linestyle='dotted', linewidth=1)
            prev_y = y
