from mlrl.common.cython._types cimport float32, uint32


cdef extern from "boosting/binning/label_binning_equal_width.hpp" namespace "boosting" nogil:

    cdef cppclass IEqualWidthLabelBinningConfig:

        # Functions:

        float32 getBinRatio() const

        IEqualWidthLabelBinningConfig& setBinRatio(float32 binRatio) except +

        uint32 getMinBins() const

        IEqualWidthLabelBinningConfig& setMinBins(uint32 minBins) except +

        uint32 getMaxBins() const

        IEqualWidthLabelBinningConfig& setMaxBins(uint32 maxBins) except +


cdef class EqualWidthLabelBinningConfig:

    # Attributes:

    cdef IEqualWidthLabelBinningConfig* config_ptr
