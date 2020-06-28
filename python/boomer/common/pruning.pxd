# distutils: language=c++
from boomer.common._arrays cimport intp, uint32
from boomer.common.rules cimport Condition
from boomer.common.rule_induction cimport IndexedArray
from boomer.common.losses cimport Loss
from boomer.common.head_refinement cimport HeadRefinement

from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map as map


cdef class Pruning:

    # Functions:

    cdef pair[uint32[::1], uint32] prune(self, map[intp, IndexedArray*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, uint32[::1] covered_examples_mask,
                                         uint32 covered_examples_target, uint32[::1] weights, intp[::1] label_indices,
                                         Loss loss, HeadRefinement head_refinement)


cdef class IREP(Pruning):

    # Functions:

    cdef pair[uint32[::1], uint32] prune(self, map[intp, IndexedArray*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, uint32[::1] covered_examples_mask,
                                         uint32 covered_examples_target, uint32[::1] weights, intp[::1] label_indices,
                                         Loss loss, HeadRefinement head_refinement)
