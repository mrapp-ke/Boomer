from mlrl.common.cython._types cimport uint32

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/input/nominal_feature_mask.hpp" nogil:

    cdef cppclass INominalFeatureMask:
        pass


cdef extern from "common/input/nominal_feature_mask_equal.hpp" nogil:

    cdef cppclass IEqualNominalFeatureMask(INominalFeatureMask):
        pass


    unique_ptr[IEqualNominalFeatureMask] createEqualNominalFeatureMask(bool nominal)


cdef extern from "common/input/nominal_feature_mask_mixed.hpp" nogil:

    cdef cppclass IMixedNominalFeatureMask(INominalFeatureMask):

        # Functions:

        void setNominal(uint32 featureIndex, bool nominal)


    unique_ptr[IMixedNominalFeatureMask] createMixedNominalFeatureMask(uint32 numFeatures)


cdef class NominalFeatureMask:

    # Functions:

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self)


cdef class EqualNominalFeatureMask(NominalFeatureMask):

    # Attributes:

    cdef unique_ptr[IEqualNominalFeatureMask] nominal_feature_mask_ptr


cdef class MixedNominalFeatureMask(NominalFeatureMask):

    # Attributes:

    cdef unique_ptr[IMixedNominalFeatureMask] nominal_feature_mask_ptr
