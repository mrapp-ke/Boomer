"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of models. The characteristics can be written to one or several
outputs, e.g., to the console or to a file.
"""
import logging as log

from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np

from mlrl.common.cython.rule_model import CompleteHead, ConjunctiveBody, EmptyBody, PartialHead, RuleModel, \
    RuleModelVisitor
from mlrl.common.learners import Learner
from mlrl.common.options import Options

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import format_float, format_percentage, format_table
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType


class ModelCharacteristicsWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow to write the characteristics of a model to one or several sinks.
    """

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write the characteristics of a model to the console.
        """

        def __init__(self):
            super().__init__(title='Model characteristics')

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write the characteristics of a model to CSV files.
        """

        def __init__(self, output_dir: str):
            super().__init__(output_dir=output_dir, file_name='model_characteristics')

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)


class RuleModelCharacteristicsWriter(ModelCharacteristicsWriter):
    """
    Allows to write the characteristics of a `RuleModel` to one or several sinks.
    """

    class RuleModelCharacteristics(Formattable, Tabularizable):
        """
        Stores the characteristics of a `RuleModel`.
        """

        def __init__(self, default_rule_index: int, default_rule_pos_predictions: int,
                     default_rule_neg_predictions: int, num_leq: np.ndarray, num_gr: np.ndarray, num_eq: np.ndarray,
                     num_neq: np.ndarray, num_pos_predictions: np.ndarray, num_neg_predictions: np.ndarray):
            """
            :param default_rule_index:              The index of the default rule or None, if no default rule is used
            :param default_rule_pos_predictions:    The number of positive predictions of the default rule, if any
            :param default_rule_neg_predictions:    The number of negative predictions of the default rule, if any
            :param num_leq:                         A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                    conditions that use the <= operator per rule
            :param num_gr:                          A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                    conditions that use the > operator per rule
            :param num_eq:                          A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                    conditions that use the == operator per rule
            :param num_neq:                         A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                    conditions that use the != operator per rule
            :param num_pos_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                    positive predictions per rule
            :param num_neg_predictions:             A `np.ndarray`, shape `(num_rules)` that stores the number of
                                                    negative predictions per rule
            """
            self.default_rule_index = default_rule_index
            self.default_rule_pos_predictions = default_rule_pos_predictions
            self.default_rule_neg_predictions = default_rule_neg_predictions
            self.num_leq = num_leq
            self.num_gr = num_gr
            self.num_eq = num_eq
            self.num_neq = num_neq
            self.num_pos_predictions = num_pos_predictions
            self.num_neg_predictions = num_neg_predictions

        def format(self, _: Options, **kwargs):
            num_predictions = self.num_pos_predictions + self.num_neg_predictions
            num_conditions = self.num_leq + self.num_gr + self.num_eq + self.num_neq
            num_total_conditions = np.sum(num_conditions)

            if num_total_conditions > 0:
                frac_leq = np.sum(self.num_leq) / num_total_conditions * 100
                frac_gr = np.sum(self.num_gr) / num_total_conditions * 100
                frac_eq = np.sum(self.num_eq) / num_total_conditions * 100
                frac_neq = np.sum(self.num_neq) / num_total_conditions * 100
                num_conditions_mean = np.mean(num_conditions)
                num_conditions_min = np.min(num_conditions)
                num_conditions_max = np.max(num_conditions)

            else:
                frac_leq = 0.0
                frac_gr = 0.0
                frac_eq = 0.0
                frac_neq = 0.0
                num_conditions_mean = 0.0
                num_conditions_min = 0.0
                num_conditions_max = 0.0

            num_total_predictions = np.sum(num_predictions)

            if num_total_predictions > 0:
                frac_pos = np.sum(self.num_pos_predictions) / num_total_predictions * 100
                frac_neg = np.sum(self.num_neg_predictions) / num_total_predictions * 100
                num_predictions_mean = np.mean(num_predictions)
                num_predictions_min = np.min(num_predictions)
                num_predictions_max = np.max(num_predictions)
            else:
                frac_pos = 0.0
                frac_neg = 0.0
                num_predictions_mean = 0.0
                num_predictions_min = 0.0
                num_predictions_max = 0.0

            num_rules = num_predictions.shape[0]

            header = ['Statistics about conditions', 'Total', '<= operator', '> operator', '== operator', '!= operator']
            alignment = ['left', 'right', 'right', 'right', 'right', 'right']
            rows = []

            if self.default_rule_index is not None:
                rows.append([
                    'Default rule',
                    str(0),
                    format_percentage(0),
                    format_percentage(0),
                    format_percentage(0),
                    format_percentage(0)
                ])

            rows.append([
                str(num_rules) + ' local rules',
                str(num_total_conditions),
                format_percentage(frac_leq),
                format_percentage(frac_gr),
                format_percentage(frac_eq),
                format_percentage(frac_neq)
            ])
            text = format_table(rows, header=header, alignment=alignment) + '\n\n'

            header = ['Statistics about predictions', 'Total', 'Positive', 'Negative']
            alignment = ['left', 'right', 'right', 'right']
            rows = []

            if self.default_rule_index is not None:
                default_rule_num_predictions = self.default_rule_pos_predictions + self.default_rule_neg_predictions
                default_rule_frac_pos = self.default_rule_pos_predictions / default_rule_num_predictions * 100
                default_rule_frac_neg = self.default_rule_neg_predictions / default_rule_num_predictions * 100
                rows.append([
                    'Default rule',
                    str(default_rule_num_predictions),
                    format_percentage(default_rule_frac_pos),
                    format_percentage(default_rule_frac_neg)
                ])

            rows.append([
                str(num_rules) + ' local rules',
                str(num_total_predictions),
                format_percentage(frac_pos),
                format_percentage(frac_neg)
            ])
            text += format_table(rows, header=header, alignment=alignment) + '\n\n'

            header = ['Statistics per local rule', 'Minimum', 'Average', 'Maximum']
            rows = []
            rows.append([
                'Conditions',
                format_float(num_conditions_min),
                format_float(num_conditions_mean),
                format_float(num_conditions_max)
            ])
            rows.append([
                'Predictions',
                format_float(num_predictions_min),
                format_float(num_predictions_mean),
                format_float(num_predictions_max)
            ])
            return text + format_table(rows, header=header)

        def tabularize(self, _: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
            rows = []
            default_rule_index = self.default_rule_index
            num_rules = len(self.num_pos_predictions)
            num_total_rules = num_rules if default_rule_index is None else num_rules + 1
            n = 0

            for i in range(num_total_rules):
                rule_name = 'Rule ' + str(i + 1)

                if i == default_rule_index:
                    rule_name += ' (Default rule)'
                    num_leq = 0
                    num_gr = 0
                    num_eq = 0
                    num_neq = 0
                    num_pos_predictions = self.default_rule_pos_predictions
                    num_neg_predictions = self.default_rule_neg_predictions
                else:
                    num_leq = self.num_leq[n]
                    num_gr = self.num_gr[n]
                    num_eq = self.num_eq[n]
                    num_neq = self.num_neq[n]
                    num_pos_predictions = self.num_pos_predictions[n]
                    num_neg_predictions = self.num_neg_predictions[n]
                    n += 1

                num_numerical = num_leq + num_gr
                num_nominal = num_eq + num_neq
                num_conditions = num_numerical + num_nominal
                num_predictions = num_pos_predictions + num_neg_predictions
                rows.append({
                    'Rule': rule_name,
                    'conditions': num_conditions,
                    'numerical conditions': num_numerical,
                    'conditions using <= operator': num_leq,
                    'conditions using > operator': num_gr,
                    'nominal conditions': num_nominal,
                    'conditions using == operator': num_eq,
                    'conditions using != operator': num_neq,
                    'predictions': num_predictions,
                    'pos. predictions': num_pos_predictions,
                    'neg. predictions': num_neg_predictions
                })

            return rows

    class RuleModelCharacteristicsVisitor(RuleModelVisitor):
        """
        A visitor that allows to determine the characteristics of a `RuleModel`.
        """

        def __init__(self):
            self.num_leq = []
            self.num_gr = []
            self.num_eq = []
            self.num_neq = []
            self.num_pos_predictions = []
            self.num_neg_predictions = []
            self.default_rule_index = None
            self.default_rule_pos_predictions = 0
            self.default_rule_neg_predictions = 0
            self.index = -1

        def visit_empty_body(self, _: EmptyBody):
            self.index += 1
            self.default_rule_index = self.index

        def visit_conjunctive_body(self, body: ConjunctiveBody):
            self.index += 1
            self.num_leq.append(body.leq_indices.shape[0] if body.leq_indices is not None else 0)
            self.num_gr.append(body.gr_indices.shape[0] if body.gr_indices is not None else 0)
            self.num_eq.append(body.eq_indices.shape[0] if body.eq_indices is not None else 0)
            self.num_neq.append(body.neq_indices.shape[0] if body.neq_indices is not None else 0)

        def visit_complete_head(self, head: CompleteHead):
            num_pos_predictions = np.count_nonzero(head.scores > 0)
            num_neg_predictions = head.scores.shape[0] - num_pos_predictions

            if self.index == self.default_rule_index:
                self.default_rule_pos_predictions = num_pos_predictions
                self.default_rule_neg_predictions = num_neg_predictions
            else:
                self.num_pos_predictions.append(num_pos_predictions)
                self.num_neg_predictions.append(num_neg_predictions)

        def visit_partial_head(self, head: PartialHead):
            num_pos_predictions = np.count_nonzero(head.scores > 0)
            num_neg_predictions = head.scores.shape[0] - num_pos_predictions

            if self.index == self.default_rule_index:
                self.default_rule_pos_predictions = num_pos_predictions
                self.default_rule_neg_predictions = num_neg_predictions
            else:
                self.num_pos_predictions.append(num_pos_predictions)
                self.num_neg_predictions.append(num_neg_predictions)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        if isinstance(learner, Learner):
            model = learner.model_

            if isinstance(model, RuleModel):
                visitor = RuleModelCharacteristicsWriter.RuleModelCharacteristicsVisitor()
                model.visit_used(visitor)
                return RuleModelCharacteristicsWriter.RuleModelCharacteristics(
                    default_rule_index=visitor.default_rule_index,
                    default_rule_pos_predictions=visitor.default_rule_pos_predictions,
                    default_rule_neg_predictions=visitor.default_rule_neg_predictions,
                    num_leq=np.asarray(visitor.num_leq),
                    num_gr=np.asarray(visitor.num_gr),
                    num_eq=np.asarray(visitor.num_eq),
                    num_neq=np.asarray(visitor.num_neq),
                    num_pos_predictions=np.asarray(visitor.num_pos_predictions),
                    num_neg_predictions=np.asarray(visitor.num_neg_predictions))

        log.error('The learner does not support to obtain model characteristics')
        return None
