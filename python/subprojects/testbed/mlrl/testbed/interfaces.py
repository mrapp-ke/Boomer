#!/usr/bin/python


"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides common interfaces that are implemented by several classes.
"""
from abc import ABC


class Randomized(ABC):
    """
    A base class for all classifiers, rankers or modules that use RNGs.

    Attributes
        random_state   The seed to be used by RNGs
    """

    random_state: int = 1
