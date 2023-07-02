"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing probability calibration models. The models can be written to one or several outputs, e.g.,
to the console or to a file.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mlrl.common.cython.probability_calibration import IsotonicProbabilityCalibrationModel, \
    IsotonicProbabilityCalibrationModelVisitor, NoProbabilityCalibrationModel
from mlrl.common.options import Options
from mlrl.common.rule_learners import RuleLearner

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import OPTION_DECIMALS, format_float, format_table
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType


class ProbabilityCalibrationModelWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that allow to write textual representations of probability calibration models
    to one or several sinks.
    """

    class IsotonicProbabilityCalibrationModelFormattable(IsotonicProbabilityCalibrationModelVisitor, Formattable,
                                                         Tabularizable):
        """
        Allows to create a textual representation of a model for the calibration of probabilities via isotonic
        regression.
        """

        def __init__(self, calibration_model: IsotonicProbabilityCalibrationModel, list_title: str):
            """
            :param calibration_model: The probability calibration model
            :param list_title:        The title of an individual list that is contained by the calibration model
            """
            self.calibration_model = calibration_model
            self.list_title = list_title
            self.bins: Dict[int, List[Tuple[float, float]]] = {}

        def __format_threshold_column(self, list_index: int) -> str:
            return self.list_title + ' ' + str(list_index + 1) + ' thresholds'

        def __format_probability_column(self, list_index: int) -> str:
            return self.list_title + ' ' + str(list_index + 1) + ' probabilities'

        def visit_bin(self, list_index: int, threshold: float, probability: float):
            bin_list = self.bins.setdefault(list_index, [])
            bin_list.append((threshold, probability))

        def format(self, options: Options, **kwargs) -> str:
            self.calibration_model.visit(self)
            decimals = options.get_int(OPTION_DECIMALS, 4)
            bins = self.bins
            result = ''

            for list_index in sorted(bins.keys()):
                header = [self.__format_threshold_column(list_index), self.__format_probability_column(list_index)]
                rows = []

                for threshold, probability in bins[list_index]:
                    rows.append(
                        [format_float(threshold, decimals=decimals),
                         format_float(probability, decimals=decimals)])

                if len(result) > 0:
                    result += '\n'

                result += format_table(rows, header=header)

            return result

        def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
            self.calibration_model.visit(self)
            decimals = options.get_int(OPTION_DECIMALS, 0)
            bins = self.bins
            rows = []
            end = False
            i = 0

            while not end:
                columns = {}
                end = True

                for list_index in bins.keys():
                    bin_list = bins[list_index]
                    column_probability = self.__format_probability_column(list_index)
                    column_threshold = self.__format_threshold_column(list_index)

                    if len(bin_list) > i:
                        probability, threshold = bin_list[i]
                        columns[column_probability] = format_float(probability, decimals=decimals)
                        columns[column_threshold] = format_float(threshold, decimals=decimals)
                        end = False
                    else:
                        columns[column_probability] = None
                        columns[column_threshold] = None

                if not end:
                    rows.append(columns)

                i += 1

            return rows

    class NoProbabilityCalibrationModelFormattable(Formattable, Tabularizable):
        """
        Allows to create a textual representation of a model for the calibration of probabilities that does not make any
        adjustments.
        """

        def format(self, options: Options, **kwargs) -> str:
            return 'No calibration model used'

        def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
            return None

    def __init__(self, sinks: List[OutputWriter.Sink], list_title: str):
        """
        :param list_title: The title of an individual list that is contained by a calibration model
        """
        super().__init__(sinks)
        self.list_title = list_title

    @abstractmethod
    def _get_calibration_model(self, learner: RuleLearner) -> Any:
        """
        Must be implemented by subclasses in order to retrieve the calibration model from a rule learner.

        :param learner: The rule learner
        :return:        The calibration model
        """
        pass

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        if isinstance(learner, RuleLearner):
            calibration_model = self._get_calibration_model(learner)

            if isinstance(calibration_model, IsotonicProbabilityCalibrationModel):
                return ProbabilityCalibrationModelWriter.IsotonicProbabilityCalibrationModelFormattable(
                    calibration_model=calibration_model, list_title=self.list_title)
            elif isinstance(calibration_model, NoProbabilityCalibrationModel):
                return ProbabilityCalibrationModelWriter.NoProbabilityCalibrationModelFormattable()

        log.error('The learner does not support to create a textual representation of the calibration model')
        return None


class MarginalProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allow to write textual representations of models for the calibration of marginal probabilities to one or several
    sinks.
    """

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write textual representations of models for the calibration of marginal probabilities to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Marginal probability calibration model', options=options)

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write textual representations of models for the calibration of marginal probabilities to a CSV file.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='marginal_probability_calibration_model', options=options)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks, list_title='Label')

    def _get_calibration_model(self, learner: RuleLearner) -> Any:
        return learner.marginal_probability_calibration_model_


class JointProbabilityCalibrationModelWriter(ProbabilityCalibrationModelWriter):
    """
    Allow to write textual representations of models for the calibration of joint probabilities to one or several sinks.
    """

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write textual representations of models for the calibration of joint probabilities to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Joint probability calibration model', options=options)

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write textual representations of models for the calibration of joint probabilities to a CSV file.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='joint_probability_calibration_model', options=options)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks, list_title='Label vector')

    def _get_calibration_model(self, learner: RuleLearner) -> Any:
        return learner.joint_probability_calibration_model_
