from mlrl.common.cython._types cimport float32

from libcpp.memory cimport unique_ptr


cdef extern from "common/sampling/instance_sampling.hpp" nogil:

    cdef cppclass IInstanceSamplingFactory:
        pass


cdef extern from "common/sampling/instance_sampling_with_replacement.hpp" nogil:

    cdef cppclass InstanceSamplingWithReplacementFactoryImpl"InstanceSamplingWithReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_without_replacement.hpp" nogil:

    cdef cppclass InstanceSamplingWithoutReplacementFactoryImpl"InstanceSamplingWithoutReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithoutReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedSamplingFactoryImpl"LabelWiseStratifiedSamplingFactory"(IInstanceSamplingFactory):

        # Constructors:

        LabelWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_stratified_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseStratifiedSamplingFactoryImpl"ExampleWiseStratifiedSamplingFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        ExampleWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_no.hpp" nogil:

    cdef cppclass NoInstanceSamplingFactoryImpl"NoInstanceSamplingFactory"(IInstanceSamplingFactory):
        pass


cdef class InstanceSamplingFactory:

    # Attributes:

    cdef unique_ptr[IInstanceSamplingFactory] instance_sampling_factory_ptr


cdef class InstanceSamplingWithReplacementFactory(InstanceSamplingFactory):
    pass


cdef class InstanceSamplingWithoutReplacementFactory(InstanceSamplingFactory):
    pass


cdef class LabelWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    pass


cdef class ExampleWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    pass


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    pass
