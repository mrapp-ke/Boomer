from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport Prediction
from boomer.common._tuples cimport IndexedFloat32Array
from boomer.common.rules cimport Condition
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.head_refinement cimport HeadRefinement

from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map


cdef class Pruning:

    # Functions:

    cdef pair[uint32[::1], uint32] prune(self, unordered_map[uint32, IndexedFloat32Array*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, Prediction* head,
                                         uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                         uint32[::1] weights, AbstractStatistics* statistics,
                                         HeadRefinement head_refinement)


cdef class IREP(Pruning):

    # Functions:

    cdef pair[uint32[::1], uint32] prune(self, unordered_map[uint32, IndexedFloat32Array*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, Prediction* head,
                                         uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                         uint32[::1] weights, AbstractStatistics* statistics,
                                         HeadRefinement head_refinement)
