# distutils: language=c++
from boomer.algorithm._arrays cimport intp, uint32, float32, float64
from boomer.algorithm._losses cimport Loss
from boomer.algorithm._head_refinement cimport HeadRefinement
from boomer.algorithm._utils cimport s_condition

from libcpp.list cimport list


cdef class Pruning:

    # Functions:

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, HeadRefinement head_refinement,
                       intp[::1] covered_example_indices, intp[::1] label_indices)

    cdef intp[::1] prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, list[s_condition] conditions)


cdef class IREP(Pruning):

    # Attributes:

    cdef float64 original_quality_score

    cdef intp[::1] label_indices

    cdef intp[::1] covered_example_indices

    cdef Loss loss

    cdef HeadRefinement head_refinement

    cdef uint32[::1] weights

    # Functions:

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, HeadRefinement head_refinement,
                       intp[::1] covered_example_indices, intp[::1] label_indices)

    cdef intp[::1] prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, list[s_condition] conditions)
