# distutils: language=c++
from boomer.common._arrays cimport intp, uint8, float32
from boomer.common._random cimport RNG
from boomer.common.rules cimport ModelBuilder
from boomer.common.losses cimport Loss
from boomer.common.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.common.pruning cimport Pruning
from boomer.common.shrinkage cimport Shrinkage
from boomer.common.head_refinement cimport HeadRefinement

from libcpp.unordered_map cimport unordered_map as map


"""
A struct that stores a value of type float32 and a corresponding index that refers to the (original) position of the
value in an array.
"""
cdef struct IndexedValue:
    intp index
    float32 value


"""
A struct that contains a pointer to a C-array of type `IndexedValue`. The attribute `num_elements` specifies how many
elements the array contains.
"""
cdef struct IndexedArray:
    IndexedValue* data
    intp num_elements


"""
A struct that contains a pointer to a struct of type `IndexedArray`, representing the indices and feature values of the
training examples that are covered by a rule. The attribute `num_conditions` specifies how many conditions the rule
contained when the array was updated for the last time. It may be used to check if the array is still valid or must be
updated.
"""
cdef struct IndexedArrayWrapper:
    IndexedArray* array
    intp num_conditions


cdef class FeatureMatrix:

    # Attributes:

    cdef readonly intp num_examples

    cdef readonly intp num_features

    # Functions:

    cdef IndexedArray* get_sorted_feature_values(self, intp feature_index)


cdef class DenseFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef float32[::1, :] x

    # Functions:

    cdef IndexedArray* get_sorted_feature_values(self, intp feature_index)


cdef class SparseFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef float32[::1] x_data

    cdef intp[::1] x_row_indices

    cdef intp[::1] x_col_indices

    # Functions:

    cdef IndexedArray* get_sorted_feature_values(self, intp feature_index)


cdef class RuleInduction:

    # Functions:

    cdef void induce_default_rule(self, uint8[::1, :] y, Loss loss, ModelBuilder model_builder)

    cdef bint induce_rule(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, Shrinkage shrinkage, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, RNG rng, ModelBuilder model_builder)


cdef class ExactGreedyRuleInduction(RuleInduction):

    # Attributes:

    cdef map[intp, IndexedArray*]* cache_global

    # Functions:

    cdef void induce_default_rule(self, uint8[::1, :] y, Loss loss, ModelBuilder model_builder)

    cdef bint induce_rule(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, Shrinkage shrinkage, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, RNG rng, ModelBuilder model_builder)
