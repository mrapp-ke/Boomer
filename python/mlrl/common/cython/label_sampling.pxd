from mlrl.common.cython._types cimport uint32

from libcpp.memory cimport unique_ptr


cdef extern from "common/sampling/label_sampling.hpp" nogil:

    cdef cppclass ILabelSamplingFactory:
        pass


cdef extern from "common/sampling/label_sampling_without_replacement.hpp" nogil:

    cdef cppclass LabelSamplingWithoutReplacementFactoryImpl"LabelSamplingWithoutReplacementFactory"(
            ILabelSamplingFactory):

        # Constructors:

        LabelSamplingWithoutReplacementFactoryImpl(uint32 numSamples) except +


cdef extern from "common/sampling/label_sampling_no.hpp" nogil:

    cdef cppclass NoLabelSamplingFactoryImpl"NoLabelSamplingFactory"(ILabelSamplingFactory):
        pass


cdef class LabelSamplingFactory:

    # Attributes:

    cdef unique_ptr[ILabelSamplingFactory] label_sampling_factory_ptr


cdef class LabelSamplingWithoutReplacementFactory(LabelSamplingFactory):
    pass


cdef class NoLabelSamplingFactory(LabelSamplingFactory):
    pass
