#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides a data structure that allows to store and parse options that are provided as key-value pairs.
"""
from enum import Enum
from typing import Set

from mlrl.common.strings import format_string_set, format_enum_values


class BooleanOption(Enum):
    TRUE = 'true'
    FALSE = 'false'

    @staticmethod
    def parse(s) -> bool:
        if s == BooleanOption.TRUE.value:
            return True
        elif s == BooleanOption.FALSE.value:
            return False
        raise ValueError(
            'Invalid boolean value given. Must be one of ' + format_enum_values(BooleanOption) + ', but is "' + str(
                s) + '".')


class Options:
    """
    Stores key-value pairs in a dictionary and provides methods to access and validate them.
    """

    ERROR_MESSAGE_INVALID_SYNTAX = 'Invalid syntax used to specify additional options'

    ERROR_MESSAGE_INVALID_OPTION = 'Expected comma-separated list of key-value pairs'

    def __init__(self):
        self.dict = {}

    @classmethod
    def create(cls, string: str, allowed_keys: Set[str]):
        """
        Parses the options that are provided via a given string that is formatted according to the following syntax:
        "[key1=value1,key2=value2]". If the given string is malformed, a `ValueError` will be raised.

        :param string:          The string to be parsed
        :param allowed_keys:    A set that contains all valid keys
        :return:                An object of type `Options` that stores the key-value pairs that have been parsed from
                                the given string
        """
        options = cls()

        if string is not None and len(string) > 0:
            if not string.startswith('{'):
                raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. Must start with "{", but is "'
                                 + string + '"')
            if not string.endswith('}'):
                raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. Must end with "}", but is "' + string + '"')

            string = string[1:-1]

            if len(string) > 0:
                for argument_index, argument in enumerate(string.split(',')):
                    if len(argument) > 0:
                        parts = argument.split('=')

                        if len(parts) != 2:
                            raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. '
                                             + Options.ERROR_MESSAGE_INVALID_OPTION + ', but got element "' + argument
                                             + '" at index ' + str(argument_index))

                        key = parts[0]

                        if len(key) == 0:
                            raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. '
                                             + Options.ERROR_MESSAGE_INVALID_OPTION
                                             + ', but key is missing from element "' + argument + '" at index '
                                             + str(argument_index))

                        if key not in allowed_keys:
                            raise ValueError('Key must be one of ' + format_string_set(allowed_keys) + ', but got key "'
                                             + key + '" at index ' + str(argument_index))

                        value = parts[1]

                        if len(value) == 0:
                            raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. '
                                             + Options.ERROR_MESSAGE_INVALID_OPTION
                                             + ', but value is missing from element "' + argument + '" at index '
                                             + str(argument_index))

                        options.dict[key] = value

        return options

    def get_string(self, key: str, default_value: str) -> str:
        """
        Returns a string that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dict:
            return str(self.dict[key])

        return default_value

    def get_bool(self, key: str, default_value: bool) -> bool:
        """
        Returns a boolean that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dict:
            value = str(self.dict[key])
            return BooleanOption.parse(value)

        return default_value

    def get_int(self, key: str, default_value: int) -> int:
        """
        Returns an integer that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dict:
            value = self.dict[key]

            try:
                value = int(value)
            except ValueError:
                raise ValueError('Value for key "' + key + '" is expected to be an integer, but is "' + str(value)
                                 + '"')

            return value

        return default_value

    def get_float(self, key: str, default_value: float) -> float:
        """
        Returns a float that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                THe value that is associated with the given key or the given default value
        """
        if key in self.dict:
            value = self.dict[key]

            try:
                value = float(value)
            except ValueError:
                raise ValueError('Value for key "' + key + '" is expected to be a float, but is "' + str(value) + '"')

            return value

        return default_value
