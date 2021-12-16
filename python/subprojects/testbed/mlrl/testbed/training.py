#!/usr/bin/python


"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating multi-label classifiers using either cross validation or separate training
and test sets.
"""
import logging as log
import os.path as path
from abc import ABC, abstractmethod
from timeit import default_timer as timer

from mlrl.testbed.data import MetaData, load_data_set_and_meta_data, load_data_set, one_hot_encode
from mlrl.testbed.interfaces import Randomized
from mlrl.testbed.io import SUFFIX_ARFF, SUFFIX_XML, get_file_name
from sklearn.model_selection import KFold


class DataSet:
    """
    Stores the properties of a data set to be used for training and evaluating multi-label classifiers.
    """

    def __init__(self, data_dir: str, data_set_name: str, use_one_hot_encoding: bool):
        """
        :param data_dir:                The path of the directory where the data set is located
        :param data_set_name:           The name of the data set
        :param use_one_hot_encoding:    True, if one-hot-encoding should be used to encode nominal attributes, False
                                        otherwise
        """
        self.data_dir = data_dir
        self.data_set_name = data_set_name
        self.use_one_hot_encoding = use_one_hot_encoding


class CrossValidation(Randomized, ABC):
    """
    A base class for all classes that use cross validation or a train-test split to train and evaluate a multi-label
    classifier or ranker.
    """

    def __init__(self, data_set: DataSet, num_folds: int, current_fold: int):
        """
        :param data_set:        The properties of the data set to be used
        :param num_folds:       The total number of folds to be used by cross validation or 1, if separate training and
                                test sets should be used
        :param current_fold:    The cross validation fold to be performed or -1, if all folds should be performed
        """
        self.data_set = data_set
        self.num_folds = num_folds
        self.current_fold = current_fold

    def run(self):
        start_time = timer()
        num_folds = self.num_folds

        if num_folds > 1:
            self.__cross_validate(num_folds)
        else:
            self.__train_test_split()

        end_time = timer()
        run_time = end_time - start_time
        log.info('Successfully finished after %s seconds', run_time)

    def __cross_validate(self, num_folds: int):
        """
        Performs n-fold cross validation.

        :param num_folds: The total number of cross validation folds
        """
        current_fold = self.current_fold
        log.info('Performing ' + (
            'full' if current_fold < 0 else ('fold ' + str(current_fold + 1) + ' of')) + ' %s-fold cross validation...',
                 num_folds)
        data_set = self.data_set
        data_set_name = data_set.data_set_name
        x, y, meta_data = load_data_set_and_meta_data(data_set.data_dir, get_file_name(data_set_name, SUFFIX_ARFF),
                                                      get_file_name(data_set_name, SUFFIX_XML))

        if data_set.use_one_hot_encoding:
            x, _, meta_data = one_hot_encode(x, y, meta_data)

        # Cross validate
        if current_fold < 0:
            first_fold = 0
            last_fold = num_folds - 1
        else:
            first_fold = current_fold
            last_fold = current_fold

        i = 0
        k_fold = KFold(n_splits=num_folds, random_state=self.random_state, shuffle=True)

        for train_indices, test_indices in k_fold.split(x, y):
            if current_fold < 0 or i == current_fold:
                log.info('Fold %s / %s:', (i + 1), num_folds)

                # Create training set for current fold
                train_x = x[train_indices]
                train_y = y[train_indices]

                # Create test set for current fold
                test_x = x[test_indices]
                test_y = y[test_indices]

                # Train & evaluate classifier
                self._train_and_evaluate(meta_data, train_indices, train_x, train_y, test_indices, test_x, test_y,
                                         first_fold=first_fold, current_fold=i, last_fold=last_fold,
                                         num_folds=num_folds)

            i += 1

    def __train_test_split(self):
        """
        Trains the classifier used in the experiment on a training set and validates it on a test set.
        """

        log.info('Using separate training and test sets...')

        # Load training data
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        use_one_hot_encoding = data_set.use_one_hot_encoding
        train_arff_file_name = get_file_name(data_set_name + '-train', SUFFIX_ARFF)
        train_arff_file = path.join(data_dir, train_arff_file_name)
        test_data_exists = True

        if not path.isfile(train_arff_file):
            train_arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)
            log.warning('File \'' + train_arff_file + '\' does not exist. Using \'' +
                        path.join(data_dir, train_arff_file_name) + '\' instead!')
            test_data_exists = False

        train_x, train_y, meta_data = load_data_set_and_meta_data(data_dir, train_arff_file_name,
                                                                  get_file_name(data_set_name, SUFFIX_XML))

        if use_one_hot_encoding:
            train_x, encoder, meta_data = one_hot_encode(train_x, train_y, meta_data)
        else:
            encoder = None

        # Load test data
        if test_data_exists:
            test_x, test_y = load_data_set(data_dir, get_file_name(data_set_name + '-test', SUFFIX_ARFF), meta_data)

            if encoder is not None:
                test_x, _ = one_hot_encode(test_x, test_y, meta_data, encoder=encoder)
        else:
            log.warning('No test data set available. Model will be evaluated on the training data!')
            test_x = train_x
            test_y = train_y

        # Train and evaluate classifier
        self._train_and_evaluate(meta_data, None, train_x, train_y, None, test_x, test_y, first_fold=0,
                                 current_fold=0, last_fold=0, num_folds=1)

    @abstractmethod
    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        """
        The function that is invoked to build a multi-label classifier or ranker on a training set and evaluate it on a
        test set.

        :param meta_data:       The meta data of the training data set
        :param train_indices:   The indices of the training examples or None, if no cross validation is used
        :param train_x:         The feature matrix of the training examples
        :param train_y:         The label matrix of the training examples
        :param test_indices:    The indices of the test examples or None, if no cross validation is used
        :param test_x:          The feature matrix of the test examples
        :param test_y:          The label matrix of the test examples
        :param first_fold:      The first fold or 0, if no cross validation is used
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param last_fold:       The last fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        pass
