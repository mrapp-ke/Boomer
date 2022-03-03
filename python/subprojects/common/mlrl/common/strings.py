#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with strings.
"""
from functools import reduce
from typing import Set, Dict


def format_enum_values(enum) -> str:
    """
    Creates and returns a textual representation of an enum's values.

    :param enum:    The enum to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b.value + '"', enum, '') + '}'


def format_string_set(strings: Set[str]) -> str:
    """
    Creates and returns a textual representation of the strings in a set.

    :param strings: The set of strings to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b + '"', strings, '') + '}'


def format_dict_keys(dictionary: Dict[str, Set[str]]) -> str:
    """
    Creates and returns a textual representation of the keys in a dictionary.

    :param dictionary:  The dictionary to be formatted
    :return:            The textual representation that has been created
    """
    return format_string_set(set(dictionary.keys()))
