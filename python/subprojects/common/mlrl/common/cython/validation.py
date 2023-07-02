"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


def assert_greater(name: str, value, threshold):
    """
    Raises a `ValueError` if a given value is not greater than a specific threshold.

    :param name:        The name of the parameter, the value  corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value <= threshold:
        raise ValueError('Invalid value given for parameter "' + name + '": Must be greater than ' + str(threshold)
                         + ', but is ' + str(value))


def assert_greater_or_equal(name: str, value, threshold):
    """
    Raises a `ValueError` if a given value is not greater or equal to a specific threshold.

    :param name:        The name of the parameter, the value corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value < threshold:
        raise ValueError('Invalid value given for parameter "' + name + '": Must be greater or equal to '
                         + str(threshold) + ', but is ' + str(value))


def assert_less(name: str, value, threshold):
    """
    Raises a `ValueError` if a given value is not less than a specific threshold.
    
    :param name:        The name of the parameter, the value corresponds to 
    :param value:       The value
    :param threshold:   The threshold
    """
    if value >= threshold:
        raise ValueError('Invalid value given for parameter "' + name + '": Must be less than ' + str(threshold)
                         + ', but is ' + str(value))


def assert_less_or_equal(name: str, value, threshold):
    """
    Raises a `ValueError` if a given value is not less or equal to a specific threshold.

    :param name:        The name of the parameter, the value corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value > threshold:
        raise ValueError('Invalid value given for parameter "' + name + '": Must be less or equal to ' + str(threshold)
                         + ', but is ' + str(value))


def assert_multiple(name: str, value, other):
    """
    Raises a `ValueError` if a given value is not a multiple of another value.

    :param name:    The name of the parameter, the value corresponds to
    :param value:   The value that should be a multiple of `other`
    :param other:   The other value
    """
    if value % other != 0:
        raise ValueError('Invalid value given for parameter "' + name + '": Must be a multiple of ' + str(other)
                         + ', but is ' + str(value))


def assert_not_none(name: str, value):
    """
    Raises a `ValueError` if a given value is None.

    :param name:    The name of the parameter, the value corresponds to
    :param value:   The value
    """
    if value is None:
        raise ValueError('Invalid value given for parameter "' + name + '": Must not be None')
