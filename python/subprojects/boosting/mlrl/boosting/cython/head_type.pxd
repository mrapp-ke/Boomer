from mlrl.common.cython._types cimport float32, uint32


cdef extern from "boosting/rule_evaluation/head_type_partial_fixed.hpp" namespace "boosting" nogil:

    cdef cppclass IFixedPartialHeadConfig:

        # Functions:

        float32 getLabelRatio() const

        IFixedPartialHeadConfig& setLabelRatio(float32 labelRatio) except +

        uint32 getMinLabels() const

        IFixedPartialHeadConfig& setMinLabels(uint32 minLabels) except +

        uint32 getMaxLabels() const

        IFixedPartialHeadConfig& setMaxLabels(uint32 maxLabels) except +


cdef extern from "boosting/rule_evaluation/head_type_partial_dynamic.hpp" namespace "boosting" nogil:

    cdef cppclass IDynamicPartialHeadConfig:

        # Functions:

        float32 getThreshold() const

        IDynamicPartialHeadConfig& setThreshold(float32 threshold) except +

        float32 getExponent() const

        IDynamicPartialHeadConfig& setExponent(float32 exponent) except +


cdef class FixedPartialHeadConfig:

    # Attributes:

    cdef IFixedPartialHeadConfig* config_ptr


cdef class DynamicPartialHeadConfig:

    # Attributes:

    cdef IDynamicPartialHeadConfig* config_ptr
