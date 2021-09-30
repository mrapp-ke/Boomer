from mlrl.common.cython._types cimport float32
from mlrl.common.cython.instance_sampling cimport IInstanceSamplingFactory, InstanceSamplingFactory


cdef extern from "seco/sampling/instance_sampling_with_replacement.hpp" namespace "seco" nogil:

    cdef cppclass InstanceSamplingWithReplacementFactoryImpl"seco::InstanceSamplingWithReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "seco/sampling/instance_sampling_without_replacement.hpp" namespace "seco" nogil:

    cdef cppclass InstanceSamplingWithoutReplacementFactoryImpl"seco::InstanceSamplingWithoutReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithoutReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "seco/sampling/instance_sampling_no.hpp" namespace "seco" nogil:

    cdef cppclass NoInstanceSamplingFactoryImpl"seco::NoInstanceSamplingFactory"(IInstanceSamplingFactory):
        pass


cdef class InstanceSamplingWithReplacementFactory(InstanceSamplingFactory):
    pass


cdef class InstanceSamplingWithoutReplacementFactory(InstanceSamplingFactory):
    pass


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    pass
