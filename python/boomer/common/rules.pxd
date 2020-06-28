# distutils: language=c++
from boomer.common._arrays cimport uint8, uint32, intp, float32, float64

from libcpp.list cimport list as double_linked_list


"""
An enum that specifies all possible types of operators used by a condition of a rule.
"""
cdef enum Comparator:
    LEQ = 0
    GR = 1
    EQ = 2
    NEQ = 3


"""
A struct that represents a condition of a rule. It consists of the index of the feature the condition corresponds to,
the type of the operator that is used by the condition, as well as a threshold.
"""
cdef struct Condition:
    intp feature_index
    Comparator comparator
    float32 threshold


cdef class Body:

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class EmptyBody(Body):

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class ConjunctiveBody(Body):

    # Attributes:

    cdef readonly intp[::1] leq_feature_indices

    cdef readonly float32[::1] leq_thresholds

    cdef readonly intp[::1] gr_feature_indices

    cdef readonly float32[::1] gr_thresholds

    cdef readonly intp[::1] eq_feature_indices

    cdef readonly float32[::1] eq_thresholds

    cdef readonly intp[::1] neq_feature_indices

    cdef readonly float32[::1] neq_thresholds

    # Functions:

    cdef bint covers(self, float32[::1] example)

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n)


cdef class Head:

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class FullHead(Head):

    # Attributes:

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class PartialHead(Head):

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] scores

    # Functions:

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask=*)


cdef class Rule:

    # Attributes:

    cdef readonly Body body

    cdef readonly Head head

    # Functions:

    cdef predict(self, float32[:, ::1] x, float64[:, ::1] predictions, uint8[:, ::1] mask=*)

    cdef predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices, intp num_features,
                     float32[::1] tmp_array1, uint32[::1] tmp_array2, uint32 n, float64[:, ::1] predictions,
                     uint8[:, ::1] mask=*)


cdef class RuleModel:

    # Functions:

    cdef void add_rule(self, Rule rule)

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels)

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                                     intp num_features, intp num_labels)


cdef class RuleList(RuleModel):

    # Attributes:

    cdef readonly bint use_mask

    cdef readonly list rules

    # Functions:

    cdef void add_rule(self, Rule rule)

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels)

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                                     intp num_features, intp num_labels)


cdef class ModelBuilder:

    # Functions:

    cdef void set_default_rule(self, float64[::1] scores)

    cdef void add_rule(self, intp[::1] label_indices, float64[::1] scores, double_linked_list[Condition] conditions,
                       intp[::1] num_conditions_per_comparator)

    cdef RuleModel build_model(self)


cdef class RuleListBuilder(ModelBuilder):

    # Attributes:

    cdef readonly bint use_mask

    cdef readonly bint default_rule_at_end

    cdef RuleList rule_list

    cdef Rule default_rule

    # Functions:

    cdef void set_default_rule(self, float64[::1] scores)

    cdef void add_rule(self, intp[::1] label_indices, float64[::1] scores, double_linked_list[Condition] conditions,
                       intp[::1] num_conditions_per_comparator)

    cdef RuleModel build_model(self)
