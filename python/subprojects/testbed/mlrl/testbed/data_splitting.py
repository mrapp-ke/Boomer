"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating multi-label classifiers using either cross validation or separate training
and test sets.
"""
import logging as log
import os.path as path

from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from timeit import default_timer as timer
from typing import List, Optional

from scipy.sparse import vstack
from sklearn.model_selection import KFold, train_test_split

from mlrl.testbed.data import MetaData, load_data_set, load_data_set_and_meta_data, one_hot_encode
from mlrl.testbed.format import format_duration
from mlrl.testbed.io import SUFFIX_ARFF, SUFFIX_XML, get_file_name, get_file_name_per_fold


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


class DataSplit(ABC):
    """
    Provides information about a split of the available data that is used for training and testing.
    """

    @abstractmethod
    def is_train_test_separated(self) -> bool:
        """
        Returns whether the training data is separated from the test data or not.

        :return: True, if the training data is separated from the test data, False otherwise
        """
        pass

    @abstractmethod
    def get_num_folds(self) -> int:
        """
        Returns the total number of cross validation folds.

        :return: The total number of cross validation folds or 1, if no cross validation is used
        """
        pass

    @abstractmethod
    def get_fold(self) -> Optional[int]:
        """
        Returns the cross validation fold, this split corresponds to.

        :return: The cross validation fold, starting at 0, or None, if no cross validation is used
        """
        pass

    @abstractmethod
    def is_last_fold(self) -> bool:
        """
        Returns whether this split corresponds to the last fold of a cross validation or not.

        :return: True, if this split corresponds to the last fold, False otherwise
        """
        pass

    def is_cross_validation_used(self) -> bool:
        """
        Returns whether cross validation is used or not.

        :return: True, if cross validation is used, False otherwise
        """
        return self.get_num_folds() > 1


class NoSplit(DataSplit):
    """
    Provides information about data that has not been split into separate training and test data.
    """

    def is_train_test_separated(self) -> bool:
        return False

    def get_num_folds(self) -> int:
        return 1

    def get_fold(self) -> Optional[int]:
        return None

    def is_last_fold(self) -> bool:
        return True


class TrainingTestSplit(DataSplit):
    """
    Provides information about a split of the available data into training and test data.
    """

    def is_train_test_separated(self) -> bool:
        return True

    def get_num_folds(self) -> int:
        return 1

    def get_fold(self) -> Optional[int]:
        return None

    def is_last_fold(self) -> bool:
        return True


class CrossValidationFold(DataSplit):
    """
    Provides information about a split of the available data that is used by a single fold of a cross validation.
    """

    def __init__(self, num_folds: int, fold: int, current_fold: int):
        """
        :param num_folds:       The total number of folds
        :param fold:            The fold, starting at 0
        :param current_fold:    The cross validation fold to be performed or -1, if all folds are performed
        """
        self.num_folds = num_folds
        self.fold = fold
        self.current_fold = current_fold

    def is_train_test_separated(self) -> bool:
        return True

    def get_num_folds(self) -> int:
        return self.num_folds

    def get_fold(self) -> Optional[int]:
        return self.fold

    def is_last_fold(self) -> bool:
        return self.current_fold < 0 and self.fold == self.num_folds - 1


class CrossValidationOverall(DataSplit):
    """
    Provides information about the overall splits of a cross validation.
    """

    def __init__(self, num_folds: int):
        """
        :param num_folds: The total number of folds
        """
        self.num_folds = num_folds

    def is_train_test_separated(self) -> bool:
        return True

    def get_num_folds(self) -> int:
        return self.num_folds

    def get_fold(self) -> Optional[int]:
        return None

    def is_last_fold(self) -> bool:
        return True


class DataType(Enum):
    """
    Characterizes data as either training or test data.
    """
    TRAINING = 'training'
    TEST = 'test'

    def get_file_name(self, name: str) -> str:
        """
        Returns a file name that corresponds to a specific type of data.

        :param name:    The name of the file (without suffix)
        :return:        The file name
        """
        return name + '_' + str(self.value)


class DataSplitter(ABC):
    """
    An abstract base class for all classes that split a data set into training and test data.
    """

    class Callback(ABC):
        """
        An abstract base class for all classes that train and evaluate a model given a predefined split of the available
        data.
        """

        @abstractmethod
        def train_and_evaluate(self, meta_data: MetaData, data_split: DataSplit, train_x, train_y, test_x, test_y):
            """
            The function that is invoked to build a multi-label classifier or ranker on a training set and evaluate it
            on a test set.

            :param meta_data:   The meta-data of the training data set
            :param data_split:  Information about the split of the available data that should be used for building and
                                evaluating a classifier or ranker
            :param train_x:     The feature matrix of the training examples
            :param train_y:     The label matrix of the training examples
            :param test_x:      The feature matrix of the test examples
            :param test_y:      The label matrix of the test examples
            """
            pass

    def run(self, callback: Callback):
        """
        :param callback: The callback that should be used for training and evaluating models
        """
        start_time = timer()
        self._split_data(callback)
        end_time = timer()
        run_time = end_time - start_time
        log.info('Successfully finished after %s', format_duration(run_time))

    @abstractmethod
    def _split_data(self, callback: Callback):
        """
        Must be implemented by subclasses in order to split the available data.

        :param callback: The callback that should be used for training and evaluating models
        """
        pass


def check_if_files_exist(directory: str, file_names: List[str]) -> bool:
    missing_files = []

    for file_name in file_names:
        file = path.join(directory, file_name)

        if not path.isfile(file):
            missing_files.append(file)

    num_missing_files = len(missing_files)

    if num_missing_files == 0:
        return True
    elif num_missing_files == len(file_names):
        return False
    else:
        raise IOError('The following files do not exist: '
                      + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b + '"', missing_files, ''))


class NoSplitter(DataSplitter):
    """
    Does not split the available data into separate train and test sets.
    """

    def __init__(self, data_set: DataSet):
        """
        :param data_set: The properties of the data set to be used
        """
        self.data_set = data_set

    def _split_data(self, callback: DataSplitter.Callback):
        log.warning('Not using separate training and test sets. The model will be evaluated on the training data...')
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        use_one_hot_encoding = data_set.use_one_hot_encoding

        # Load data set...
        arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)
        xml_file_name = get_file_name(data_set_name, SUFFIX_XML)
        x, y, meta_data = load_data_set_and_meta_data(data_dir, arff_file_name, xml_file_name)

        # Apply one-hot-encoding, if necessary...
        if use_one_hot_encoding:
            x, _, encoded_meta_data = one_hot_encode(x, y, meta_data)
        else:
            encoded_meta_data = None

        # Train and evaluate classifier...
        data_split = NoSplit()
        callback.train_and_evaluate(encoded_meta_data if encoded_meta_data is not None else meta_data, data_split, x, y,
                                    x, y)


class TrainTestSplitter(DataSplitter):
    """
    Splits the available data into a single train and test set.
    """

    def __init__(self, data_set: DataSet, test_size: float, random_state: int):
        """
        :param data_set:        The properties of the data set to be used
        :param test_size:       The fraction of the available data to be used as the test set
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        self.data_set = data_set
        self.test_size = test_size
        self.random_state = random_state

    def _split_data(self, callback: DataSplitter.Callback):
        log.info('Using separate training and test sets...')
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        use_one_hot_encoding = data_set.use_one_hot_encoding
        xml_file_name = get_file_name(data_set_name, SUFFIX_XML)

        # Check if ARFF files with predefined training and test data are available...
        train_arff_file_name = get_file_name(DataType.TRAINING.get_file_name(data_set_name), SUFFIX_ARFF)
        test_arff_file_name = get_file_name(DataType.TEST.get_file_name(data_set_name), SUFFIX_ARFF)
        predefined_split = check_if_files_exist(data_dir, [train_arff_file_name, test_arff_file_name])

        if not predefined_split:
            train_arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)

        # Load (training) data set...
        train_x, train_y, meta_data = load_data_set_and_meta_data(data_dir, train_arff_file_name, xml_file_name)

        # Apply one-hot-encoding, if necessary...
        if use_one_hot_encoding:
            train_x, encoder, encoded_meta_data = one_hot_encode(train_x, train_y, meta_data)
        else:
            encoder = None
            encoded_meta_data = None

        if predefined_split:
            # Load test data set...
            test_x, test_y = load_data_set(data_dir, test_arff_file_name, meta_data)

            # Apply one-hot-encoding, if necessary...
            if encoder is not None:
                test_x, _, _ = one_hot_encode(test_x, test_y, meta_data, encoder=encoder)
        else:
            # Split data set into training and test data...
            train_x, test_x, train_y, test_y = train_test_split(train_x,
                                                                train_y,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                shuffle=True)

        # Train and evaluate classifier...
        data_split = TrainingTestSplit()
        callback.train_and_evaluate(encoded_meta_data if encoded_meta_data is not None else meta_data, data_split,
                                    train_x, train_y, test_x, test_y)


class CrossValidationSplitter(DataSplitter):
    """
    Splits the available data into training and test sets corresponding to the individual folds of a cross validation.
    """

    def __init__(self, data_set: DataSet, num_folds: int, current_fold: int, random_state: int):
        """
        :param data_set:        The properties of the data set to be used
        :param num_folds:       The total number of folds to be used by cross validation or 1, if separate training and
                                test sets should be used
        :param current_fold:    The cross validation fold to be performed or -1, if all folds should be performed
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        self.data_set = data_set
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.random_state = random_state

    def _split_data(self, callback: DataSplitter.Callback):
        num_folds = self.num_folds
        current_fold = self.current_fold
        log.info(
            'Performing ' + ('full' if current_fold < 0 else
                             ('fold ' + str(current_fold + 1) + ' of')) + ' %s-fold cross validation...', num_folds)
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        use_one_hot_encoding = data_set.use_one_hot_encoding
        xml_file_name = get_file_name(data_set_name, SUFFIX_XML)

        # Check if ARFF files with predefined folds are available...
        arff_file_names = [get_file_name_per_fold(data_set_name, SUFFIX_ARFF, fold) for fold in range(num_folds)]
        predefined_split = check_if_files_exist(data_dir, arff_file_names)

        if predefined_split:
            self.__predefined_cross_validation(callback,
                                               data_dir=data_dir,
                                               arff_file_names=arff_file_names,
                                               xml_file_name=xml_file_name,
                                               use_one_hot_encoding=use_one_hot_encoding,
                                               num_folds=num_folds,
                                               current_fold=current_fold)
        else:
            arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)
            self.__cross_validation(callback,
                                    data_dir=data_dir,
                                    arff_file_name=arff_file_name,
                                    xml_file_name=xml_file_name,
                                    use_one_hot_encoding=use_one_hot_encoding,
                                    num_folds=num_folds,
                                    current_fold=current_fold)

    @staticmethod
    def __predefined_cross_validation(callback: DataSplitter.Callback, data_dir: str, arff_file_names: List[str],
                                      xml_file_name: str, use_one_hot_encoding: bool, num_folds: int,
                                      current_fold: int):
        # Load first data set for the first fold...
        x, y, meta_data = load_data_set_and_meta_data(data_dir, arff_file_names[0], xml_file_name)

        # Apply one-hot-encoding, if necessary...
        if use_one_hot_encoding:
            x, encoder, encoded_meta_data = one_hot_encode(x, y, meta_data)
        else:
            encoder = None
            encoded_meta_data = None

        data = [(x, y)]

        # Load data sets for the remaining folds...
        for fold in range(1, num_folds):
            x, y = load_data_set(data_dir, arff_file_names[fold], meta_data)

            # Apply one-hot-encoding, if necessary...
            if encoder is not None:
                x, _, _ = one_hot_encode(x, y, meta_data, encoder=encoder)

            data.append((x, y))

        # Perform cross-validation...
        for fold in range(0 if current_fold < 0 else current_fold, num_folds if current_fold < 0 else current_fold + 1):
            log.info('Fold %s / %s:', (fold + 1), num_folds)

            # Create training set for current fold...
            train_x = None
            train_y = None

            for other_fold in range(num_folds):
                if other_fold != fold:
                    x, y = data[other_fold]

                    if train_x is None:
                        train_x = x
                        train_y = y
                    else:
                        train_x = vstack((train_x, x))
                        train_y = vstack((train_y, y))

            # Obtain test set for current fold...
            test_x, test_y = data[fold]

            # Train and evaluate classifier...
            data_split = CrossValidationFold(num_folds=num_folds, fold=fold, current_fold=current_fold)
            callback.train_and_evaluate(encoded_meta_data if encoded_meta_data is not None else meta_data, data_split,
                                        train_x, train_y, test_x, test_y)

    def __cross_validation(self, callback: DataSplitter.Callback, data_dir: str, arff_file_name: str,
                           xml_file_name: str, use_one_hot_encoding: bool, num_folds: int, current_fold: int):
        # Load data set...
        x, y, meta_data = load_data_set_and_meta_data(data_dir, arff_file_name, xml_file_name)

        # Apply one-hot-encoding, if necessary...
        if use_one_hot_encoding:
            x, _, encoded_meta_data = one_hot_encode(x, y, meta_data)
        else:
            encoded_meta_data = None

        # Perform cross-validation...
        k_fold = KFold(n_splits=num_folds, random_state=self.random_state, shuffle=True)

        for fold, (train_indices, test_indices) in enumerate(k_fold.split(x, y)):
            if current_fold < 0 or fold == current_fold:
                log.info('Fold %s / %s:', (fold + 1), num_folds)

                # Create training set for current fold...
                train_x = x[train_indices]
                train_y = y[train_indices]

                # Create test set for current fold...
                test_x = x[test_indices]
                test_y = y[test_indices]

                # Train and evaluate classifier...
                data_split = CrossValidationFold(num_folds=num_folds, fold=fold, current_fold=current_fold)
                callback.train_and_evaluate(encoded_meta_data if encoded_meta_data is not None else meta_data,
                                            data_split, train_x, train_y, test_x, test_y)
