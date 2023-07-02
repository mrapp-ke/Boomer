from mlrl.common.cython._types cimport float64


cdef extern from "boosting/rule_evaluation/regularization_manual.hpp" namespace "boosting" nogil:

    cdef cppclass IManualRegularizationConfig:

        # Functions:

        float64 getRegularizationWeight() const

        IManualRegularizationConfig& setRegularizationWeight(float64 regularizationWeight) except +


cdef class ManualRegularizationConfig:

    # Attributes:

    cdef IManualRegularizationConfig* config_ptr
