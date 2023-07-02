from mlrl.common.cython._types cimport float32, uint32


cdef extern from "common/binning/feature_binning_equal_width.hpp" nogil:

    cdef cppclass IEqualWidthFeatureBinningConfig:

        # Functions:

        IEqualWidthFeatureBinningConfig& setBinRatio(float32 binRatio) except +

        float32 getBinRatio() const

        IEqualWidthFeatureBinningConfig& setMinBins(uint32 minBins) except +

        uint32 getMinBins() const

        IEqualWidthFeatureBinningConfig& setMaxBins(uint32 maxBins) except +

        uint32 getMaxBins() const


cdef extern from "common/binning/feature_binning_equal_frequency.hpp" nogil:

    cdef cppclass IEqualFrequencyFeatureBinningConfig:

        # Functions:

        IEqualFrequencyFeatureBinningConfig& setBinRatio(float32 binRatio) except +

        float32 getBinRatio() const

        IEqualFrequencyFeatureBinningConfig& setMinBins(uint32 minBins) except +

        uint32 getMinBins() const

        IEqualFrequencyFeatureBinningConfig& setMaxBins(uint32 maxBins) except +

        uint32 getMaxBins() const


cdef class EqualWidthFeatureBinningConfig:

    # Attributes:

    cdef IEqualWidthFeatureBinningConfig* config_ptr


cdef class EqualFrequencyFeatureBinningConfig:

    # Attributes:

    cdef IEqualFrequencyFeatureBinningConfig* config_ptr
