#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for representing the model learned by a classifier or ranker.
"""
from typing import List

import numpy as np
from boomer.algorithm._model import Rule

DTYPE_INTP = np.intp

DTYPE_UINT8 = np.uint8

DTYPE_FLOAT32 = np.float32

DTYPE_FLOAT64 = np.float64


# Type alias for a theory, which is a list containing several rules
Theory = List[Rule]
