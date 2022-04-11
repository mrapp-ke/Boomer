#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for saving/loading models to/from disk.
"""
import _pickle as pickle
import logging as log
import os.path as path


class ModelPersistence:
    """
    Allows to save a model in a file and load it later.
    """

    def __init__(self, model_dir: str):
        """
        :param model_dir: The path of the directory where models should be saved
        """
        self.model_dir = model_dir

    def save_model(self, model, model_name: str, fold: int = None):
        """
        Saves a model to a file.

        :param model:       The model to be persisted
        :param model_name:  The name of the model to be persisted
        :param fold:        The fold, the model corresponds to, or None if no cross validation is used
        """

        file_path = path.join(self.model_dir, ModelPersistence.__get_file_name(model_name, fold))
        log.debug('Saving model to file \"%s\"...', file_path)

        try:
            with open(file_path, mode='wb') as output_stream:
                pickle.dump(model, output_stream, -1)
                log.info('Successfully saved model to file \"%s\"', file_path)
        except IOError:
            log.exception('Failed to save model to file \"%s\"', file_path)

    def load_model(self, model_name: str, fold: int = None, raise_exception: bool = False):
        """
        Loads a model from a file.

        :param model_name:      The name of the model to be loaded
        :param fold:            The fold, the model corresponds to, or None if no cross validation is used
        :param raise_exception: True, if an exception should be raised if an error occurs, False, if None should be
                                returned in such case
        :return:                The loaded model
        """

        file_path = path.join(self.model_dir, ModelPersistence.__get_file_name(model_name, fold))
        log.debug("Loading model from file \"%s\"...", file_path)

        try:
            with open(file_path, mode='rb') as input_stream:
                model = pickle.load(input_stream)
                log.info('Successfully loaded model from file \"%s\"', file_path)
                return model
        except IOError as e:
            log.error('Failed to load model from file \"%s\"', file_path)

            if raise_exception:
                raise e
            else:
                return None

    @staticmethod
    def __get_file_name(model_name: str, fold: int):
        """
        Returns the name of the file that is used to persist a model.

        :param model_name:          The name of the model to be persisted
        :param fold:                The fold, the model corresponds to, or None if no cross validation is used
        :return:                    The name of the file
        """
        file_name = model_name

        if fold is not None:
            file_name += ('_fold-' + str(fold + 1))

        return file_name + '.model'
