#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides commonly used utility functions.
"""
import numpy as np
from boomer.algorithm._model import Rule, Body, EmptyBody, ConjunctiveBody, Head, FullHead, PartialHead

from boomer.stats import Stats


def format_rule(stats: Stats, rule: Rule) -> str:
    """
    Formats a specific rule as a text.

    :param stats:   Statistics about the training data set
    :param rule:    The rule to be formatted
    :return:        The text
    """
    text = __print_body(rule.body)
    text += ' -> '
    text += __print_head(stats, rule.head)
    return text


def __print_body(body: Body) -> str:
    if isinstance(body, EmptyBody):
        return '{}'
    elif isinstance(body, ConjunctiveBody):
        return '{' + __print_conjunctive_body(body) + '}'
    else:
        raise ValueError('Body has unknown type: ' + type(body).__name__)


def __print_conjunctive_body(body: ConjunctiveBody) -> str:
    text = __print_conditions(np.asarray(body.leq_feature_indices), np.asarray(body.leq_thresholds))
    return __print_conditions(np.asarray(body.gr_feature_indices), np.asarray(body.gr_thresholds), text)


def __print_conditions(feature_indices: np.ndarray, thresholds: np.ndarray, text: str = '') -> str:
    for i in range(feature_indices.shape[0]):
        if len(text) > 0:
            text += ' & '

        text += str(feature_indices[i])
        text += ' <= '
        text += str(thresholds[i])

    return text


def __print_head(stats: Stats, head: Head) -> str:
    if isinstance(head, FullHead):
        return '(' + __print_full_head(head) + ')'
    elif isinstance(head, PartialHead):
        return '(' + __print_partial_head(stats, head) + ')'
    else:
        raise ValueError('Head has unknown type: ' + type(head).__name__)


def __print_full_head(head: FullHead) -> str:
    text = ''
    scores = np.asarray(head.scores)

    for i in range(scores.shape[0]):
        if len(text) > 0:
            text += ', '

        text += '{0:.2f}'.format(scores[i])

    return text


def __print_partial_head(stats: Stats, head: PartialHead) -> str:
    text = ''
    scores = np.asarray(head.scores)
    label_indices = np.asarray(head.label_indices)

    for i in range(stats.num_labels):
        if len(text) > 0:
            text += ', '

        l = np.argwhere(label_indices == i)

        if np.size(l) > 0:
            text += '{0:.2f}'.format(scores[l.item()])
        else:
            text += '?'

    return text
