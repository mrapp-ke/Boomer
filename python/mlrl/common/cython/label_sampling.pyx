"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_unique


cdef class LabelSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelSamplingFactory`.
    """
    pass


cdef class LabelSamplingWithoutReplacementFactory(LabelSamplingFactory):
    """
    A wrapper for the C++ class `LabelSamplingWithoutReplacementFactory`.
    """

    def __cinit__(self, uint32 num_samples):
        """
        :param num_samples: The number of labels to be included in the sample
        """
        self.label_sampling_factory_ptr = <unique_ptr[ILabelSamplingFactory]>make_unique[LabelSamplingWithoutReplacementFactoryImpl](
            num_samples)


cdef class NoLabelSamplingFactory(LabelSamplingFactory):
    """
    A wrapper for the C++ class `NoLabelSamplingFactory`.
    """

    def __cinit__(self):
        self.label_sampling_factory_ptr = <unique_ptr[ILabelSamplingFactory]>make_unique[NoLabelSamplingFactoryImpl]()
