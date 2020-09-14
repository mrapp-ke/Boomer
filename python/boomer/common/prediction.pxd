from boomer.common._arrays cimport uint32, float32, float64
from boomer.common.rules cimport RuleModel


cdef class Predictor:

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, uint32 num_labels, RuleModel model)

    cpdef object predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                             uint32 num_features, uint32 num_labels, RuleModel model)


cdef class DensePredictor(Predictor):

    # Attributes:

    cdef readonly TransformationFunction transformation_function

    # Functions:

    cpdef object predict(self, float32[:, ::1] x, uint32 num_labels, RuleModel model)

    cpdef object predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                             uint32 num_features, uint32 num_labels, RuleModel model)

cdef class TransformationFunction:

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)


cdef class SignFunction(TransformationFunction):

    # Functions:

    cdef object transform_matrix(self, float64[:, ::1] m)
