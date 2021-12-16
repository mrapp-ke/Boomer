"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class FeatureSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `IFeatureSamplingFactory`.
    """
    pass


cdef class FeatureSamplingWithoutReplacementFactory(FeatureSamplingFactory):
    """
    A wrapper for the C++ class `FeatureSamplingWithoutReplacementFactory`.
    """

    def __cinit__(self, float32 sample_size):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available features). Must be in (0, 1) or 0, if the default sample size
                            `floor(log2(num_features - 1) + 1)` should be used
        """
        self.feature_sampling_factory_ptr = <unique_ptr[IFeatureSamplingFactory]>make_unique[FeatureSamplingWithoutReplacementFactoryImpl](
            sample_size)


cdef class NoFeatureSamplingFactory(FeatureSamplingFactory):
    """
    A wrapper for the C++ class `NoFeatureSamplingFactory`.
    """

    def __cinit__(self):
        self.feature_sampling_factory_ptr = <unique_ptr[IFeatureSamplingFactory]>make_unique[NoFeatureSamplingFactoryImpl]()
