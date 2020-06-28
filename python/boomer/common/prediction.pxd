from boomer.common._arrays cimport intp, float32, float64
from boomer.common.rules cimport RuleModel


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, RuleModel rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, RuleModel model)


cdef class DensePredictor(Predictor):

    # Attributes:

    cdef readonly TransformationFunction transformation_function

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, intp num_labels, RuleModel rules)

    cpdef object predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                             intp num_features, intp num_labels, RuleModel model)

cdef class TransformationFunction:

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)


cdef class SignFunction(TransformationFunction):

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)
