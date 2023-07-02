from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float64, uint32

ctypedef void (*BinVisitor)(uint32, float64, float64)


cdef extern from "common/prediction/probability_calibration_marginal.hpp" nogil:

    cdef cppclass IMarginalProbabilityCalibrationModel:
        pass        


cdef extern from "common/prediction/probability_calibration_joint.hpp" nogil:

    cdef cppclass IJointProbabilityCalibrationModel:
        pass


cdef extern from "common/prediction/probability_calibration_no.hpp" nogil:

    cdef cppclass INoProbabilityCalibrationModel(IMarginalProbabilityCalibrationModel,
                                                 IJointProbabilityCalibrationModel):
        pass

    unique_ptr[INoProbabilityCalibrationModel] createNoProbabilityCalibrationModel()


ctypedef INoProbabilityCalibrationModel* NoProbabilityCalibrationModelPtr


cdef extern from "common/prediction/probability_calibration_isotonic.hpp" nogil:

    cdef cppclass IIsotonicProbabilityCalibrationModel(IMarginalProbabilityCalibrationModel,
                                                       IJointProbabilityCalibrationModel):
        
        # Functions:

        uint32 getNumBinLists() const

        void addBin(uint32 listIndex, float64 threshold, float64 probability)

        void visit(BinVisitor) const


    unique_ptr[IIsotonicProbabilityCalibrationModel] createIsotonicProbabilityCalibrationModel(uint32 numLists)


ctypedef IIsotonicProbabilityCalibrationModel* IsotonicProbabilityCalibrationModelPtr


cdef extern from *:
    """
    #include "common/prediction/probability_calibration_isotonic.hpp"

    typedef void (*BinCythonVisitor)(void*, uint32, float64, float64);

    static inline IIsotonicProbabilityCalibrationModel::BinVisitor wrapBinVisitor(
            void* self, BinCythonVisitor visitor) {
        return [=](uint32 labelIndex, float64 threshold, float64 probability) {
            visitor(self, labelIndex, threshold, probability);
        };
    }
    """

    ctypedef void (*BinCythonVisitor)(void*, uint32, float64, float64)

    BinVisitor wrapBinVisitor(void* self, BinCythonVisitor visitor)


cdef class MarginalProbabilityCalibrationModel:

    # Functions:

    cdef IMarginalProbabilityCalibrationModel* get_marginal_probability_calibration_model_ptr(self)


cdef class JointProbabilityCalibrationModel:

    # Functions:

    cdef IJointProbabilityCalibrationModel* get_joint_probability_calibration_model_ptr(self)


cdef class NoMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[INoProbabilityCalibrationModel] probability_calibration_model_ptr

    cdef dict __dict__


cdef class NoJointProbabilityCalibrationModel(JointProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[INoProbabilityCalibrationModel] probability_calibration_model_ptr

    cdef dict __dict__


cdef class IsotonicMarginalProbabilityCalibrationModel(MarginalProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[IIsotonicProbabilityCalibrationModel] probability_calibration_model_ptr

    cdef object state

    cdef object visitor

    cdef dict __dict__

    # Functions:

    cdef __visit_bin(self, uint32 list_index, float64 threshold, float64 probability)

    cdef __serialize_bin(self, uint32 list_index, float64 threshold, float64 probability)


cdef class IsotonicJointProbabilityCalibrationModel(JointProbabilityCalibrationModel):

    # Attributes:

    cdef unique_ptr[IIsotonicProbabilityCalibrationModel] probability_calibration_model_ptr

    cdef object state

    cdef object visitor

    cdef dict __dict__

    # Functions:

    cdef __visit_bin(self, uint32 list_index, float64 threshold, float64 probability)

    cdef __serialize_bin(self, uint32 list_index, float64 threshold, float64 probability)


cdef inline MarginalProbabilityCalibrationModel create_marginal_probability_calibration_model(
        unique_ptr[IMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr):
    cdef IMarginalProbabilityCalibrationModel* ptr = marginal_probability_calibration_model_ptr.release()
    cdef INoProbabilityCalibrationModel* no_marginal_probability_calibration_model_ptr = \
        dynamic_cast[NoProbabilityCalibrationModelPtr](ptr)
    cdef NoMarginalProbabilityCalibrationModel no_marginal_probability_calibration_model
    cdef IIsotonicProbabilityCalibrationModel* isotonic_marginal_probability_calibration_model_ptr
    cdef IsotonicMarginalProbabilityCalibrationModel isotonic_marginal_probability_calibration_model

    if no_marginal_probability_calibration_model_ptr != NULL:
        no_marginal_probability_calibration_model = \
            NoMarginalProbabilityCalibrationModel.__new__(NoMarginalProbabilityCalibrationModel)
        no_marginal_probability_calibration_model.probability_calibration_model_ptr = \
            unique_ptr[INoProbabilityCalibrationModel](no_marginal_probability_calibration_model_ptr)
        return no_marginal_probability_calibration_model
    else:
        isotonic_marginal_probability_calibration_model_ptr = dynamic_cast[IsotonicProbabilityCalibrationModelPtr](ptr)
        
        if isotonic_marginal_probability_calibration_model_ptr != NULL:
            isotonic_marginal_probability_calibration_model = \
                IsotonicMarginalProbabilityCalibrationModel.__new__(IsotonicMarginalProbabilityCalibrationModel)
            isotonic_marginal_probability_calibration_model.probability_calibration_model_ptr = \
                unique_ptr[IIsotonicProbabilityCalibrationModel](isotonic_marginal_probability_calibration_model_ptr)
            return isotonic_marginal_probability_calibration_model
        else:
            del ptr
            raise RuntimeError('Encountered unsupported IMarginalProbabilityCalibrationModel object')


cdef inline JointProbabilityCalibrationModel create_joint_probability_calibration_model(
        unique_ptr[IJointProbabilityCalibrationModel] joint_probability_calibration_model_ptr):
    cdef IJointProbabilityCalibrationModel* ptr = joint_probability_calibration_model_ptr.release()
    cdef INoProbabilityCalibrationModel* no_joint_probability_calibration_model_ptr = \
        dynamic_cast[NoProbabilityCalibrationModelPtr](ptr)
    cdef NoJointProbabilityCalibrationModel no_joint_probability_calibration_model
    cdef IIsotonicProbabilityCalibrationModel* isotonic_joint_probability_calibration_model_ptr
    cdef IsotonicJointProbabilityCalibrationModel isotonic_joint_probability_calibration_model

    if no_joint_probability_calibration_model_ptr != NULL:
        no_joint_probability_calibration_model = \
            NoJointProbabilityCalibrationModel.__new__(NoJointProbabilityCalibrationModel)
        no_joint_probability_calibration_model.probability_calibration_model_ptr = \
            unique_ptr[INoProbabilityCalibrationModel](no_joint_probability_calibration_model_ptr)
        return no_joint_probability_calibration_model
    else:
        isotonic_joint_probability_calibration_model_ptr = dynamic_cast[IsotonicProbabilityCalibrationModelPtr](ptr)

        if isotonic_joint_probability_calibration_model_ptr != NULL:
            isotonic_joint_probability_calibration_model = \
                IsotonicJointProbabilityCalibrationModel.__new__(IsotonicJointProbabilityCalibrationModel)
            isotonic_joint_probability_calibration_model.probability_calibration_model_ptr = \
                unique_ptr[IIsotonicProbabilityCalibrationModel](isotonic_joint_probability_calibration_model_ptr)
            return isotonic_joint_probability_calibration_model
        else:
            del ptr
            raise RuntimeError('Encountered unsupported IJointProbabilityCalibrationModel object')
