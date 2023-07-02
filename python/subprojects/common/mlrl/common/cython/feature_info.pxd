from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32


cdef extern from "common/input/feature_info.hpp" nogil:

    cdef cppclass IFeatureInfo:
        pass


cdef extern from "common/input/feature_info_equal.hpp" nogil:

    cdef cppclass IEqualFeatureInfo(IFeatureInfo):
        pass


    unique_ptr[IEqualFeatureInfo] createOrdinalFeatureInfo()

    unique_ptr[IEqualFeatureInfo] createNominalFeatureInfo()

    unique_ptr[IEqualFeatureInfo] createNumericalFeatureInfo()


cdef extern from "common/input/feature_info_mixed.hpp" nogil:

    cdef cppclass IMixedFeatureInfo(IFeatureInfo):

        # Functions:

        void setNumerical(uint32 featureIndex)

        void setOrdinal(uint32 featureIndex)

        void setNominal(uint32 featureIndex)


    unique_ptr[IMixedFeatureInfo] createMixedFeatureInfo(uint32 numFeatures)


cdef class FeatureInfo:

    # Functions:

    cdef IFeatureInfo* get_feature_info_ptr(self)


cdef class EqualFeatureInfo(FeatureInfo):

    # Attributes:

    cdef unique_ptr[IEqualFeatureInfo] feature_info_ptr


cdef class MixedFeatureInfo(FeatureInfo):

    # Attributes:

    cdef unique_ptr[IMixedFeatureInfo] feature_info_ptr
