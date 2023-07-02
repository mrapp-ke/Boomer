"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing textual representations of models. The models can be written to one or several outputs,
e.g., to the console or to a file.
"""
import logging as log

from abc import ABC
from typing import Any, List, Optional

import numpy as np

from _io import StringIO

from mlrl.common.cython.rule_model import CompleteHead, ConjunctiveBody, EmptyBody, PartialHead, RuleModel, \
    RuleModelVisitor
from mlrl.common.learners import Learner
from mlrl.common.options import Options

from mlrl.testbed.data import Attribute, MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import format_float
from mlrl.testbed.output_writer import Formattable, OutputWriter
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType

OPTION_PRINT_FEATURE_NAMES = 'print_feature_names'

OPTION_PRINT_LABEL_NAMES = 'print_label_names'

OPTION_PRINT_NOMINAL_VALUES = 'print_nominal_values'

OPTION_PRINT_BODIES = 'print_bodies'

OPTION_PRINT_HEADS = 'print_heads'

OPTION_DECIMALS_BODY = 'decimals_body'

OPTION_DECIMALS_HEAD = 'decimals_head'


class ModelWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow to write textual representations of models to one or several
    sinks.
    """

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write textual representations of models to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Model', options=options)

    class TxtSink(OutputWriter.TxtSink):
        """
        Allows to write textual representations of models to text files.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='rules', options=options)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)


class RuleModelWriter(ModelWriter):
    """
    Allows to write textual representations of rule-based models to one or several sinks.
    """

    class RuleModelFormattable(RuleModelVisitor, Formattable):
        """
        Allows to create textual representations of the rules in a `RuleModel`.
        """

        def __init__(self, meta_data: MetaData, model: RuleModel):
            """
            :param meta_data:   The meta-data of the training data set
            :param model:       The `RuleModel`
            """
            self.attributes = meta_data.attributes
            self.labels = meta_data.labels
            self.model = model
            self.text = None
            self.print_feature_names = True
            self.print_label_names = True
            self.print_nominal_values = True
            self.print_bodies = True
            self.print_heads = True
            self.body_decimals = 2
            self.head_decimals = 2

        def visit_empty_body(self, _: EmptyBody):
            if self.print_bodies:
                self.text.write('{}')

        def __format_conditions(self, num_conditions: int, indices: np.ndarray, thresholds: np.ndarray,
                                operator: str) -> int:
            result = num_conditions

            if indices is not None and thresholds is not None:
                text = self.text
                attributes = self.attributes
                print_feature_names = self.print_feature_names
                print_nominal_values = self.print_nominal_values
                decimals = self.body_decimals

                for i in range(indices.shape[0]):
                    if result > 0:
                        text.write(' & ')

                    feature_index = indices[i]
                    threshold = thresholds[i]
                    attribute: Optional[Attribute] = attributes[feature_index] if len(
                        attributes) > feature_index else None

                    if print_feature_names and attribute is not None:
                        text.write(attribute.attribute_name)
                    else:
                        text.write(str(feature_index))

                    text.write(' ')
                    text.write(operator)
                    text.write(' ')

                    if attribute is not None and attribute.nominal_values is not None:
                        nominal_value = int(threshold)

                        if print_nominal_values and len(attribute.nominal_values) > nominal_value:
                            text.write('"' + attribute.nominal_values[nominal_value] + '"')
                        else:
                            text.write(str(nominal_value))
                    else:
                        text.write(format_float(threshold, decimals=decimals))

                    result += 1

            return result

        def visit_conjunctive_body(self, body: ConjunctiveBody):
            if self.print_bodies:
                text = self.text
                text.write('{')
                num_conditions = self.__format_conditions(0, body.leq_indices, body.leq_thresholds, '<=')
                num_conditions = self.__format_conditions(num_conditions, body.gr_indices, body.gr_thresholds, '>')
                num_conditions = self.__format_conditions(num_conditions, body.eq_indices, body.eq_thresholds, '==')
                self.__format_conditions(num_conditions, body.neq_indices, body.neq_thresholds, '!=')
                text.write('}')

        def visit_complete_head(self, head: CompleteHead):
            text = self.text

            if self.print_heads:
                print_label_names = self.print_label_names
                decimals = self.head_decimals
                labels = self.labels
                scores = head.scores

                if self.print_bodies:
                    text.write(' => ')

                text.write('(')

                for i in range(scores.shape[0]):
                    if i > 0:
                        text.write(', ')

                    if print_label_names and len(labels) > i:
                        text.write(labels[i].attribute_name)
                    else:
                        text.write(str(i))

                    text.write(' = ')
                    text.write(format_float(scores[i], decimals=decimals))

                text.write(')\n')
            elif self.print_bodies:
                text.write('\n')

        def visit_partial_head(self, head: PartialHead):
            text = self.text

            if self.print_heads:
                print_label_names = self.print_label_names
                decimals = self.head_decimals
                labels = self.labels
                indices = head.indices
                scores = head.scores

                if self.print_bodies:
                    text.write(' => ')

                text.write('(')

                for i in range(indices.shape[0]):
                    if i > 0:
                        text.write(', ')

                    label_index = indices[i]

                    if print_label_names and len(labels) > label_index:
                        text.write(labels[label_index].attribute_name)
                    else:
                        text.write(str(label_index))

                    text.write(' = ')
                    text.write(format_float(scores[i], decimals=decimals))

                text.write(')\n')
            elif self.print_bodies:
                text.write('\n')

        def format(self, options: Options, **kwargs) -> str:
            self.print_feature_names = options.get_bool(OPTION_PRINT_FEATURE_NAMES, True)
            self.print_label_names = options.get_bool(OPTION_PRINT_LABEL_NAMES, True)
            self.print_nominal_values = options.get_bool(OPTION_PRINT_NOMINAL_VALUES, True)
            self.print_bodies = options.get_bool(OPTION_PRINT_BODIES, True)
            self.print_heads = options.get_bool(OPTION_PRINT_HEADS, True)
            self.body_decimals = options.get_int(OPTION_DECIMALS_BODY, 2)
            self.head_decimals = options.get_int(OPTION_DECIMALS_HEAD, 2)
            self.text = StringIO()
            self.model.visit_used(self)
            text = self.text.getvalue()
            self.text.close()
            return text

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        if isinstance(learner, Learner):
            model = learner.model_

            if isinstance(model, RuleModel):
                return RuleModelWriter.RuleModelFormattable(meta_data, model)

        log.error('The learner does not support to create a textual representation of the model')
        return None
