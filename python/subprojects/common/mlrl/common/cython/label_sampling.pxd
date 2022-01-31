from mlrl.common.cython._types cimport uint32


cdef extern from "common/sampling/label_sampling_without_replacement.hpp" nogil:

    cdef cppclass ILabelSamplingWithoutReplacementConfig:

        # Functions:

        uint32 getNumSamples() const

        ILabelSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples) except +


cdef class LabelSamplingWithoutReplacementConfig:

    # Attributes:

    cdef ILabelSamplingWithoutReplacementConfig* config_ptr
