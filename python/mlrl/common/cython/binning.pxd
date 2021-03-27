from mlrl.common.cython._types cimport uint32, float32

from libcpp.memory cimport shared_ptr


cdef extern from "common/binning/feature_binning.hpp" nogil:

    cdef cppclass IFeatureBinning:
        pass


cdef extern from "common/binning/feature_binning_equal_frequency.hpp" nogil:

    cdef cppclass EqualFrequencyFeatureBinningImpl"EqualFrequencyFeatureBinning"(IFeatureBinning):

        # Constructors:

        EqualFrequencyFeatureBinningImpl(float32 binRatio, uint32 minBins, uint32 maxBins) except +


cdef extern from "common/binning/feature_binning_equal_width.hpp" nogil:

    cdef cppclass EqualWidthFeatureBinningImpl"EqualWidthFeatureBinning"(IFeatureBinning):

        # Constructors:

        EqualWidthFeatureBinningImpl(float32 binRatio, uint32 minBins, uint32 maxBins) except +


cdef class FeatureBinning:

    # Attributes:

    cdef shared_ptr[IFeatureBinning] binning_ptr


cdef class EqualFrequencyFeatureBinning(FeatureBinning):
    pass


cdef class EqualWidthFeatureBinning(FeatureBinning):
    pass
