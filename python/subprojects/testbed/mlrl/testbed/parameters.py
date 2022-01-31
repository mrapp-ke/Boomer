#!/usr/bin/python

"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for parameter tuning.
"""
import logging as log
from abc import ABC
from abc import abstractmethod

from mlrl.testbed.data import MetaData
from mlrl.testbed.interfaces import Randomized
from mlrl.testbed.io import clear_directory
from mlrl.testbed.io import open_readable_csv_file, create_csv_dict_writer
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_reader
from mlrl.testbed.training import CrossValidation, DataSet
from sklearn.model_selection import KFold


class ParameterSearch(Randomized, ABC):
    """
    A base class for all classes that implement strategies to search for optimal parameters given a training data set.
    """

    @abstractmethod
    def search(self, meta_data: MetaData, x, y, first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        """
        Tests different parameter settings given a training data set.

        :param meta_data:       The meta data of the training data set
        :param x:               The feature matrix of the training examples
        :param y:               The label matrix of the training examples
        :param first_fold:      The first fold or 0, if no cross validation is used
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param last_fold:       The last fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns the best parameter setting tested so far.

        :return: A dictionary that stores the parameters
        """
        pass

    @abstractmethod
    def get_score(self):
        """
        Returns the evaluation score that has been achieved using the best parameter setting.

        :return: An evaluation score
        """


class NestedCrossValidation(ParameterSearch):
    """
    Allows to search for optimal parameters using (nested) cross validation.

    """

    def __init__(self, num_nested_folds: int):
        """
        :param num_nested_folds: The total number of folds to be used by the (nested) cross validation
        """
        self.num_nested_folds = num_nested_folds

    def search(self, meta_data: MetaData, x, y, first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        num_nested_folds = self.num_nested_folds
        random_state = self.random_state
        first_nested_fold = 0
        last_nested_fold = num_nested_folds - 1
        i = 0
        k_fold = KFold(n_splits=num_nested_folds, random_state=random_state, shuffle=True)

        for train, test in k_fold.split(x, y):
            log.info('Nested fold %s / %s:', (i + 1), num_nested_folds)

            # Create training set for current fold
            train_x = x[train]
            train_y = y[train]

            # Create test set for current fold
            test_x = x[test]
            test_y = y[test]

            self._test_parameters(meta_data, train_x, train_y, test_x, test_y, first_outer_fold=first_fold,
                                  current_outer_fold=current_fold, last_outer_fold=last_fold, num_outer_folds=num_folds,
                                  first_nested_fold=first_nested_fold, current_nested_fold=i,
                                  last_nested_fold=last_nested_fold, num_nested_folds=num_nested_folds)
            i += 1

    @abstractmethod
    def _test_parameters(self, meta_data: MetaData, train_x, train_y, test_x, test_y, first_outer_fold: int,
                         current_outer_fold: int, last_outer_fold: int, num_outer_folds: int, first_nested_fold: int,
                         current_nested_fold: int, last_nested_fold: int, num_nested_folds: int):
        """
        Must be implemented by subclasses in order to test different parameter settings on a given training data set and
        evaluate them on a test data set.

        :param meta_data:       The meta data of the training data set
        :param train_x:                The feature matrix of the training examples
        :param train_y:                The label matrix of the training examples
        :param test_x:                 The feature matrix of the test examples
        :param test_y:                 The label matrix of the test examples
        :param first_outer_fold:       The first (outer) fold or 0, if no cross validation is used
        :param current_outer_fold:     The current (outer) fold starting at 0, or 0 if no cross validation is used
        :param last_outer_fold:        The last (outer) fold or 0, if no cross validation is used
        :param num_outer_folds:        The total number of (outer) cross validation folds or 1, if no cross validation
                                       is used
        :param first_nested_fold:      The first (nested) fold or 0, if no cross validation is used
        :param current_nested_fold:    The current (nested) fold starting at 0, or 0 if no cross validation is used
        :param last_nested_fold:       The last (nested) fold or 0, if no cross validation is used
        :param num_nested_folds:       The total number of (nested) cross validation folds or 1, if no cross validation
                                       is used
        """
        pass


class ParameterInput(ABC):

    @abstractmethod
    def read_parameters(self, fold: int = None) -> dict:
        """
        Reads a parameter setting from the input.

        :param fold:    The fold, the parameter setting corresponds to, or None, if the parameter setting does not
                        correspond to a specific fold
        :return:        A dictionary that stores the parameters
        """
        pass


class ParameterCsvInput(ParameterInput):
    """
    Reads parameter settings from CSV files.
    """

    def __init__(self, input_dir: str):
        """
        :param input_dir: The path of the directory, the CSV files should be read from
        """
        self.input_dir = input_dir

    def read_parameters(self, fold: int = None) -> dict:
        with open_readable_csv_file(self.input_dir, 'parameters', fold) as csv_file:
            csv_reader = create_csv_dict_reader(csv_file)
            return dict(next(csv_reader))


class ParameterOutput(ABC):
    """
    An abstract base class for all outputs, parameter settings may be written to.
    """

    @abstractmethod
    def write_parameters(self, parameters: dict, score: float, total_folds: int, fold: int = None):
        """
        Writes a parameter setting to the output.

        :param parameters:  A dictionary that stores the parameters
        :param score:       The evaluation score that has been achieved using the parameter setting
        :param total_folds: The total number of folds
        :param fold:        The fold, the parameter setting corresponds to, or None, if the parameter setting does not
                            correspond to a specific fold
        """
        pass


class ParameterLogOutput(ParameterOutput):
    """
    Outputs parameter settings using the logger.
    """

    def write_parameters(self, parameters: dict, score: float, total_folds: int, fold: int = None):
        param_text = ''

        for parameter, value in parameters.items():
            if len(param_text) > 0:
                param_text += '\n'

            param_text += (parameter + ': ' + str(value))

        score_text = 'Evaluation score: ' + str(score)
        msg = 'Optimal parameter setting' + ('' if fold is None else ' (Fold ' + str(fold + 1) + ')') + ':\n\n%s\n%s\n'
        log.info(msg, param_text, score_text)


class ParameterCsvOutput(ParameterOutput):
    """
    Writes parameter settings to CSV files.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_parameters(self, parameters: dict, score: float, total_folds: int, fold: int = None):
        header = parameters.keys()

        with open_writable_csv_file(self.output_dir, 'parameters', fold) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(parameters)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class ParameterTuning(CrossValidation):
    """
    Allows to tune parameters for a single training data set or all training data sets that are used in cross validation
    using a `ParameterSearch` and writes the optimal parameters to one or several outputs.
    """

    def __init__(self, data_set: DataSet, num_folds: int, current_fold: int,
                 parameter_search: ParameterSearch, *args: ParameterOutput):
        """
        :param parameter_search:    The strategy to be used to search for optimal parameters
        :param args:                The outputs, the parameter settings should be written to
        """
        super().__init__(data_set, num_folds, current_fold)
        self.parameter_search = parameter_search
        self.outputs = args

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        parameter_search = self.parameter_search
        parameter_search.random_state = self.random_state
        parameter_search.search(meta_data, train_x, train_y, first_fold=first_fold, current_fold=current_fold,
                                last_fold=last_fold, num_folds=num_folds)
        parameters = parameter_search.get_params()
        score = parameter_search.get_score()

        for output in self.outputs:
            output.write_parameters(parameters, score, num_folds, current_fold if num_folds > 1 else None)
