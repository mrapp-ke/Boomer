"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

from abc import ABC, abstractmethod

SERIALIZATION_VERSION = 0


cdef class MarginalProbabilityCalibrationModel:
    """
    A model that may be used for the calibration of marginal probabilities.
    """

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        pass


cdef class JointProbabilityCalibrationModel:
    """
    A model that may be used for the calibration of joint probabilities.
    """

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        pass


class NoProbabilityCalibrationModel(ABC):
    """
    Defines an inteface for all models for the calibration of probabilities that do not make any adjustments.
    """
    pass


cdef class NoMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel, NoProbabilityCalibrationModel):
    """
    A model for the calibration of marginal probabilities that does not make any adjustments.
    """

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoMarginalProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoProbabilityCalibrationModel()


cdef class NoJointProbabilityCalibrationModel(JointProbabilityCalibrationModel, NoProbabilityCalibrationModel):
    """
    A model for the calibration of joint probabilities that does not make any adjustments.
    """

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    def __reduce__(self):
        return (NoJointProbabilityCalibrationModel, (), ())

    def __setstate__(self, state):
        self.probability_calibration_model_ptr = createNoProbabilityCalibrationModel()


class IsotonicProbabilityCalibrationModelVisitor(ABC):
    """
    Defines an interface for all visitors that allow to handle the bins stores by a model for the calibration of
    probabilities via isotonic regression.
    """

    @abstractmethod
    def visit_bin(self, list_index: int, threshold: float, probability: float):
        """
        Must be implemented by subclasses in order to handle a bin.

        :param list_index:  The index of the list, the bin corresponds to
        :param threshold:   The threshold of the bin
        :param probability: The probability of the bin
        """
        pass


class IsotonicProbabilityCalibrationModel(ABC):
    """
    Defines an interface for all models for the calibration of probabilities via isotonic regression.
    """

    @abstractmethod
    def visit(self, visitor: IsotonicProbabilityCalibrationModelVisitor):
        """
        Invokes a given visitor for each bin that is stored by the calibration model.

        :param visitor: The visitor to be invoked
        """
        pass


cdef class IsotonicMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel,
                                                       IsotonicProbabilityCalibrationModel):
    """
    A model for the calibration of marginal probabilities via isotonic regression.
    """

    def visit(self, visitor: IsotonicProbabilityCalibrationModelVisitor):
        self.visitor = visitor
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__visit_bin))
        self.visitor = None

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()
    
    cdef __visit_bin(self, uint32 list_index, float64 threshold, float64 probability):
        self.visitor.visit_bin(list_index, threshold, probability)

    cdef __serialize_bin(self, uint32 list_index, float64 threshold, float64 probability):
        cdef list bin_list = self.state[list_index]
        bin_list.append((threshold, probability))

    def __reduce__(self):
        cdef uint32 num_bin_lists = self.probability_calibration_model_ptr.get().getNumBinLists()
        self.state = [[] for i in range(num_bin_lists)]
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__serialize_bin))
        cdef object state = (SERIALIZATION_VERSION, self.state)
        self.state = None
        return (IsotonicMarginalProbabilityCalibrationModel, (), state)

    def __setstate__(self, state):
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError('Version of the serialized IsotonicMarginalProbabilityCalibrationModel is '
                                 + str(version) + ', expected ' + str(SERIALIZATION_VERSION))
        
        cdef list bins_per_list = state[1]
        cdef uint32 num_bin_lists = len(bins_per_list)
        cdef unique_ptr[IIsotonicProbabilityCalibrationModel] marginal_probability_calibration_model_ptr = \
            createIsotonicProbabilityCalibrationModel(num_bin_lists)
        cdef list bin_list
        cdef float64 threshold, probability
        cdef uint32 i, j, num_bins

        for i in range(num_bin_lists):
            bin_list = bins_per_list[i]
            num_bins = len(bin_list)

            for j in range(num_bins):
                threshold = bin_list[j][0]
                probability = bin_list[j][1]
                marginal_probability_calibration_model_ptr.get().addBin(i, threshold, probability)

        self.probability_calibration_model_ptr = move(marginal_probability_calibration_model_ptr)


cdef class IsotonicJointProbabilityCalibrationModel(JointProbabilityCalibrationModel,
                                                    IsotonicProbabilityCalibrationModel):
    """
    A model for the calibration of joint probabilities via isotonic regression.
    """

    def visit(self, visitor: IsotonicProbabilityCalibrationModelVisitor):
        self.visitor = visitor
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__visit_bin))
        self.visitor = None

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self):
        return self.probability_calibration_model_ptr.get()

    cdef __visit_bin(self, uint32 list_index, float64 threshold, float64 probability):
        self.visitor.visit_bin(list_index, threshold, probability)

    cdef __serialize_bin(self, uint32 list_index, float64 threshold, float64 probability):
        cdef list bin_list = self.state[list_index]
        bin_list.append((threshold, probability))

    def __reduce__(self):
        cdef uint32 num_bin_lists = self.probability_calibration_model_ptr.get().getNumBinLists()
        self.state = [[] for i in range(num_bin_lists)]
        self.probability_calibration_model_ptr.get().visit(
            wrapBinVisitor(<void*>self, <BinCythonVisitor>self.__serialize_bin))
        cdef object state = (SERIALIZATION_VERSION, self.state)
        self.state = None
        return (IsotonicJointProbabilityCalibrationModel, (), state)

    def __setstate__(self, state):
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError('Version of the serialized IsotonicJointProbabilityCalibrationModel is ' + str(version)
                                 + ', expected ' + str(SERIALIZATION_VERSION))
        
        cdef list bins_per_list = state[1]
        cdef uint32 num_bin_lists = len(bins_per_list)
        cdef unique_ptr[IIsotonicProbabilityCalibrationModel] joint_probability_calibration_model_ptr = \
            createIsotonicProbabilityCalibrationModel(num_bin_lists)
        cdef list bin_list
        cdef float64 threshold, probability
        cdef uint32 i, j, num_bins

        for i in range(num_bin_lists):
            bin_list = bins_per_list[i]
            num_bins = len(bin_list)

            for j in range(num_bins):
                threshold = bin_list[j][0]
                probability = bin_list[j][1]
                joint_probability_calibration_model_ptr.get().addBin(i, threshold, probability)

        self.probability_calibration_model_ptr = move(joint_probability_calibration_model_ptr)
