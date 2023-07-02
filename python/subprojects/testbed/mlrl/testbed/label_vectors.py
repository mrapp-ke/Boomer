"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing unique label vectors that are contained in a data set. The label vectors can be written to
one or several outputs, e.g., to the console or to a file.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scipy.sparse import lil_matrix

from mlrl.common.cython.label_space_info import LabelVectorSet, LabelVectorSetVisitor
from mlrl.common.data_types import DTYPE_UINT8
from mlrl.common.options import Options
from mlrl.common.rule_learners import RuleLearner

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import format_table
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType

OPTION_SPARSE = 'sparse'


class LabelVectorWriter(OutputWriter):
    """
    Allows to write unique label vectors that are contained in a data set to one or severals sinks.
    """

    class LabelVectors(Formattable, Tabularizable):
        """
        Stores unique label vectors that are contained in a data set.
        """

        COLUMN_INDEX = 'Index'

        COLUMN_LABEL_VECTOR = 'Label vector'

        COLUMN_FREQUENCY = 'Frequency'

        def __init__(self, num_labels: int, y=None):
            """
            :param num_labels:  The total number of available labels
            :param y:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the ground truth labels
            """
            self.num_labels = num_labels

            if y is not None:
                unique_label_vector_strings: Dict[str, int] = {}
                y = lil_matrix(y)
                separator = ','

                for label_vector in y.rows:
                    label_vector_string = separator.join(map(str, label_vector))
                    frequency = unique_label_vector_strings.setdefault(label_vector_string, 0)
                    unique_label_vector_strings[label_vector_string] = frequency + 1

                unique_label_vectors: List[Tuple[np.array, int]] = []

                for label_vector_string, frequency in unique_label_vector_strings.items():
                    label_vector = np.asarray(
                        [int(label_index) for label_index in label_vector_string.split(separator)])
                    unique_label_vectors.append((label_vector, frequency))

                self.unique_label_vectors = unique_label_vectors
            else:
                self.unique_label_vectors = []

        def __format_label_vector(self, sparse_label_vector: np.ndarray, sparse: bool) -> str:
            if sparse:
                return str(sparse_label_vector)
            else:
                dense_label_vector = np.zeros(shape=self.num_labels, dtype=DTYPE_UINT8)
                dense_label_vector[sparse_label_vector] = 1
                return str(dense_label_vector)

        def format(self, options: Options, **kwargs) -> str:
            sparse = options.get_bool(OPTION_SPARSE, False)
            header = [self.COLUMN_INDEX, self.COLUMN_LABEL_VECTOR, self.COLUMN_FREQUENCY]
            rows = []

            for i, (sparse_label_vector, frequency) in enumerate(self.unique_label_vectors):
                rows.append([i + 1, self.__format_label_vector(sparse_label_vector, sparse=sparse), frequency])

            return format_table(rows, header=header)

        def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
            sparse = options.get_bool(OPTION_SPARSE, False)
            rows = []

            for i, (sparse_label_vector, frequency) in enumerate(self.unique_label_vectors):
                columns = {
                    self.COLUMN_INDEX: i + 1,
                    self.COLUMN_LABEL_VECTOR: self.__format_label_vector(sparse_label_vector, sparse=sparse),
                    self.COLUMN_FREQUENCY: frequency
                }
                rows.append(columns)

            return rows

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write unique label vectors that are contained in a data set to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Label vectors', options=options)

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write unique label vectors that are contained in a data set to a CSV file.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='label_vectors', options=options)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        return LabelVectorWriter.LabelVectors(num_labels=y.shape[1], y=y)


class LabelVectorSetWriter(LabelVectorWriter):
    """
    Allows to write unique label vectors that are stored as part of a model learned by a rule learning algorithm to one
    or several sinks.
    """

    class Visitor(LabelVectorSetVisitor):
        """
        Allows to access the label vectors and frequencies store by a `LabelVectorSet`.
        """

        def __init__(self, num_labels: int):
            """
            :param num_labels: The total number of available labels
            """
            self.label_vectors = LabelVectorWriter.LabelVectors(num_labels=num_labels)

        def visit_label_vector(self, label_vector: np.ndarray, frequency: int):
            self.label_vectors.unique_label_vectors.append((label_vector, frequency))

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        if isinstance(learner, RuleLearner):
            label_space_info = learner.label_space_info_

            if isinstance(label_space_info, LabelVectorSet):
                visitor = LabelVectorSetWriter.Visitor(num_labels=y.shape[1])
                label_space_info.visit(visitor)
                return visitor.label_vectors

        return super()._generate_output_data(meta_data, x, y, data_split, learner, data_type, prediction_type,
                                             prediction_scope, predictions, train_time, predict_time)
