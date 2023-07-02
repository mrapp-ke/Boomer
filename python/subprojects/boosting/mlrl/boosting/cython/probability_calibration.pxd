from libcpp cimport bool


cdef extern from "boosting/prediction/probability_calibration_isotonic.hpp" namespace "boosting" nogil:

    cdef cppclass IIsotonicMarginalProbabilityCalibratorConfig:
        
        # Functions:

        bool isHoldoutSetUsed() const

        IIsotonicMarginalProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet)


    cdef cppclass IIsotonicJointProbabilityCalibratorConfig:
        
        # Functions:

        bool isHoldoutSetUsed() const

        IIsotonicJointProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet)


cdef class IsotonicMarginalProbabilityCalibratorConfig:

    # Attributes:

    cdef IIsotonicMarginalProbabilityCalibratorConfig* config_ptr


cdef class IsotonicJointProbabilityCalibratorConfig:

    # Attributes:

    cdef IIsotonicJointProbabilityCalibratorConfig* config_ptr
