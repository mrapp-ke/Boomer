"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

from mlrl.common.cython._types cimport uint32


cdef class FeatureInfo:
    """
    Provides information about the types of individual features.
    """

    cdef IFeatureInfo* get_feature_info_ptr(self):
        pass


cdef class EqualFeatureInfo(FeatureInfo):
    """
    Provides information about the types of individual features in cases where all features are of the same type, i.e.,
    where all features are either ordinal, nominal or numerical.
    """

    @classmethod
    def create_ordinal(cls) -> 'EqualFeatureInfo':
        cdef EqualFeatureInfo equal_feature_info = EqualFeatureInfo.__new__(EqualFeatureInfo)
        equal_feature_info.feature_info_ptr = createOrdinalFeatureInfo()
        return equal_feature_info

    @classmethod
    def create_nominal(cls) -> 'EqualFeatureInfo':
        cdef EqualFeatureInfo equal_feature_info = EqualFeatureInfo.__new__(EqualFeatureInfo)
        equal_feature_info.feature_info_ptr = createNominalFeatureInfo()
        return equal_feature_info

    @classmethod
    def create_numerical(cls) -> 'EqualFeatureInfo':
        cdef EqualFeatureInfo equal_feature_info = EqualFeatureInfo.__new__(EqualFeatureInfo)
        equal_feature_info.feature_info_ptr = createNumericalFeatureInfo()
        return equal_feature_info


    cdef IFeatureInfo* get_feature_info_ptr(self):
        return self.feature_info_ptr.get()


cdef class MixedFeatureInfo(FeatureInfo):
    """
    Provides information about the types of individual features in cases where different types of features, i.e.,
    ordinal, nominal and numerical ones, are available.
    """

    def __cinit__(self, uint32 num_features, list ordinal_feature_indices not None,
                  list nominal_feature_indices not None):
        """
        :param num_features:            The total number of available features
        :param ordinal_feature_indices: A list which contains the indices of all ordinal features
        :param nominal_feature_indices: A list which contains the indices of all nominal features
        """
        cdef unique_ptr[IMixedFeatureInfo] feature_info_ptr = createMixedFeatureInfo(num_features)
        cdef uint32 feature_index

        for feature_index in nominal_feature_indices:
            feature_info_ptr.get().setNominal(feature_index)

        for feature_index in ordinal_feature_indices:
            feature_info_ptr.get().setOrdinal(feature_index)

        self.feature_info_ptr = move(feature_info_ptr)

    cdef IFeatureInfo* get_feature_info_ptr(self):
        return self.feature_info_ptr.get()
