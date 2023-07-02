"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for saving/loading models to/from disk.
"""
import logging as log
import os.path as path

import _pickle as pickle

from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.io import get_file_name_per_fold

SUFFIX_MODEL = 'model'


class ModelPersistence:
    """
    Allows to save a model in a file and load it later.
    """

    def __init__(self, model_dir: str):
        """
        :param model_dir: The path of the directory where models should be saved
        """
        self.model_dir = model_dir

    def save_model(self, model, model_name: str, data_split: DataSplit):
        """
        Saves a model to a file.

        :param model:       The model to be persisted
        :param model_name:  The name of the model to be persisted
        :param data_split:  Information about the split of the available data, the model corresponds to
        """
        file_name = get_file_name_per_fold(model_name, SUFFIX_MODEL, data_split.get_fold())
        file_path = path.join(self.model_dir, file_name)
        log.debug('Saving model to file \"%s\"...', file_path)

        try:
            with open(file_path, mode='wb') as output_stream:
                pickle.dump(model, output_stream, -1)
                log.info('Successfully saved model to file \"%s\"', file_path)
        except IOError:
            log.error('Failed to save model to file \"%s\"', file_path)

    def load_model(self, model_name: str, data_split: DataSplit):
        """
        Loads a model from a file.

        :param model_name:  The name of the model to be loaded
        :param data_split:  Information about the split of the available data, the model corresponds to
        :return:            The loaded model
        """
        file_name = get_file_name_per_fold(model_name, SUFFIX_MODEL, data_split.get_fold())
        file_path = path.join(self.model_dir, file_name)
        log.debug("Loading model from file \"%s\"...", file_path)

        try:
            with open(file_path, mode='rb') as input_stream:
                model = pickle.load(input_stream)
                log.info('Successfully loaded model from file \"%s\"', file_path)
                return model
        except IOError:
            log.error('Failed to load model from file \"%s\"', file_path)
            return None
