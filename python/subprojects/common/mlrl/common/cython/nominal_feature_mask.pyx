"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move


cdef class NominalFeatureMask:
    """
    Allows to check whether individual features are nominal or not.
    """

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self):
        pass


cdef class EqualNominalFeatureMask(NominalFeatureMask):
    """
    Allows to check whether individual features are nominal or not in cases where all features are of the same type,
    i.e., where all features are either nominal or numerical/ordinal.
    """

    def __cinit__(self, bint nominal):
        """
        :param nominal: True, if all features are nominal, False, if all features are numerical/ordinal
        """
        self.nominal_feature_mask_ptr = createEqualNominalFeatureMask(nominal)


    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self):
        return self.nominal_feature_mask_ptr.get()


cdef class MixedNominalFeatureMask(NominalFeatureMask):
    """
    Allows to check whether individual features are nominal or not in cases where different types of features, i.e.,
    nominal and numerical/ordinal ones, are available.
    """

    def __cinit__(self, uint32 num_features, list nominal_feature_indices not None):
        """
        :param num_features:            The total number of available features
        :param nominal_feature_indices: A list which contains the indices of all nominal features
        """
        cdef unique_ptr[IMixedNominalFeatureMask] nominal_feature_mask_ptr = createMixedNominalFeatureMask(num_features)
        cdef uint32 i

        for i in nominal_feature_indices:
            nominal_feature_mask_ptr.get().setNominal(i, True)

        self.nominal_feature_mask_ptr = move(nominal_feature_mask_ptr)

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self):
        return self.nominal_feature_mask_ptr.get()
