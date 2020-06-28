"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement loss functions that are applied example-wise.
"""
from boomer.common._arrays cimport array_float64, fortran_matrix_float64, get_index
from boomer.boosting.differentiable_losses cimport _convert_label_into_score, _l2_norm_pow

from libc.math cimport pow, exp, fabs

from cpython.mem cimport PyMem_Malloc as malloc, PyMem_Free as free

from scipy.linalg.cython_blas cimport ddot, dspmv
from scipy.linalg.cython_lapack cimport dsysv


cdef class ExampleWiseLogisticLoss(NonDecomposableDifferentiableLoss):
    """
    A multi-label variant of the logistic loss that is applied example-wise.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the optimal
                                         scores to be predicted by rules. Increasing this value causes the model to be
                                         more conservative, setting it to 0 turns of L2 regularization entirely
        """
        self.l2_regularization_weight = l2_regularization_weight
        self.prediction = LabelIndependentPrediction.__new__(LabelIndependentPrediction)
        self.sums_of_gradients = None
        self.sums_of_hessians = None

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        # The weight to be used for L2 regularization
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of examples
        cdef intp num_examples = y.shape[0]
        # The number of labels
        cdef intp num_labels = y.shape[1]
        # A matrix that stores the expected scores for each example and label according to the ground truth
        cdef float64[::1, :] expected_scores = fortran_matrix_float64(num_examples, num_labels)
        # Pre-calculated values
        cdef float64 sum_of_exponentials = num_labels + 1
        cdef float64 sum_of_exponentials_pow = pow(sum_of_exponentials, 2)
        # Temporary variables
        cdef float64 expected_score
        cdef intp r, c, c2, i

        # We find the optimal scores to be predicted by the default rule for each label by solving a system of linear
        # equations A * X = B with one equation per label. A is a two-dimensional (symmetrical) matrix of coefficients,
        # B is an one-dimensional array of ordinates, and X is an one-dimensional array of unknowns to be determined.
        # The ordinates result from the gradients of the loss function, whereas the coefficients result from the
        # hessians. As the matrix of coefficients is symmetrical, we must only compute the hessians that correspond to
        # the upper-right triangle of the matrix and leave the remaining elements unspecified. For reasons of space
        # efficiency, we store the hessians in an one-dimensional array by appending the columns of the matrix of
        # coefficients to each other and omitting the unspecified elements.
        cdef float64[::1] ordinates = array_float64(num_labels)
        ordinates[:] = 0
        cdef intp num_hessians = __triangular_number(num_labels)  # The number of elements in the upper-right triangle
        cdef float64[::1] coefficients = array_float64(num_hessians)
        coefficients[:] = 0

        # Example-wise calculate the gradients and hessians and add them to the arrays of ordinates and coefficients...
        for r in range(num_examples):
            # Traverse the labels of the current example once to convert the ground truth labels into expected scores...
            for c in range(num_labels):
                expected_scores[r, c] = _convert_label_into_score(y[r, c])

            # Traverse the labels again to calculate the gradients and hessians...
            i = 0

            for c in range(num_labels):
                expected_score = expected_scores[r, c]

                # Calculate the first derivative (gradient) of the loss function with respect to the current label and
                # add it to the array of ordinates...
                ordinates[c] += expected_score / sum_of_exponentials

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add it to the matrix of coefficients...
                for c2 in range(c):
                    coefficients[i] -= (expected_scores[r, c2] * expected_score) / sum_of_exponentials_pow
                    i += 1

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the diagonal of the matrix of coefficients...
                coefficients[i] += (fabs(expected_score) * num_labels) / sum_of_exponentials_pow
                i += 1

        # Compute the optimal scores to be predicted by the default rule by solving the system of linear equations...
        cdef float64[::1] scores = __dsysv_float64(coefficients, ordinates, l2_regularization_weight)

        # We must traverse each example again to calculate the updated gradients and hessians based on the calculated
        # scores...
        cdef float64[::1] exponentials = ordinates # Reuse existing array instead of allocating a new one
        # A matrix that stores the gradients
        cdef float64[::1, :] gradients = fortran_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        # A matrix that stores the hessians
        cdef float64[::1, :] hessians = fortran_matrix_float64(num_examples, num_hessians)
        # An array that stores the column-wise sums of the matrix of hessians
        cdef float64[::1] total_sums_of_hessians = coefficients # Reuse existing array instead of allocating a new one
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[::1, :] current_scores = fortran_matrix_float64(num_examples, num_labels)
        # Temporary variables
        cdef float64 exponential, tmp, score

        for r in range(num_examples):
            # Traverse the labels of the current example once to create arrays of expected scores and exponentials that
            # are shared among the upcoming calculations of gradients and hessians...
            sum_of_exponentials = 1

            for c in range(num_labels):
                expected_score = expected_scores[r, c]
                exponential = exp(-expected_score * scores[c])
                exponentials[c] = exponential
                sum_of_exponentials += exponential

            sum_of_exponentials_pow = pow(sum_of_exponentials, 2)

            # Traverse the labels again to calculate the gradients and hessians...
            i = 0

            for c in range(num_labels):
                expected_score = expected_scores[r, c]
                exponential = exponentials[c]
                score = scores[c]
                current_scores[r, c] = score

                # Calculate the first derivative (gradient) of the loss function with respect to the current label and
                # add it to the matrix of gradients...
                tmp = (expected_score * exponential) / sum_of_exponentials
                # Note: The sign of the gradient is inverted (from negative to positive), because otherwise, when using
                # the sums of gradients as the ordinates for solving a system of linear equations in the function
                # `evaluate_label_dependent_predictions`, the sign must be inverted again...
                gradients[r, c] = tmp

                # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
                # each of the other labels and add them to the matrix of hessians...
                for c2 in range(c):
                    tmp = exp(-expected_scores[r, c2] * scores[c2] - expected_score * score)
                    tmp = (expected_scores[r, c2] * expected_score * tmp) / sum_of_exponentials_pow
                    hessians[r, i] = -tmp
                    i += 1

                # Calculate the second derivative (hessian) of the loss function with respect to the current label and
                # add it to the diagonal of the matrix of hessians...
                tmp = (fabs(expected_score) * exponential * (sum_of_exponentials - exponential)) / sum_of_exponentials_pow
                hessians[r, i] = tmp
                i += 1

        # Store the gradients...
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients

        # Store the hessians...
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians

        # Store the expected and currently predicted scores...
        self.expected_scores = expected_scores
        self.current_scores = current_scores

        return scores

    cdef void begin_instance_sub_sampling(self):
        # Class members
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # Reset total sums of gradients and hessians to 0...
        total_sums_of_gradients[:] = 0
        total_sums_of_hessians[:] = 0

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        # Class members
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The number of gradients/hessians...
        cdef intp num_elements = gradients.shape[1]
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # Temporary variables
        cdef intp c

        # For each label, add the gradient of the example at the given index (weighted by the given weight) to the total
        # sums of gradients...
        for c in range(num_elements):
            total_sums_of_gradients[c] += (signed_weight * gradients[example_index, c])

        # Add the hessians of the example at the given index (weighted by the given weight) to the total sums of
        # hessians...
        num_elements = hessians.shape[1]

        for c in range(num_elements):
            total_sums_of_hessians[c] += (signed_weight * hessians[example_index, c])

    cdef void begin_search(self, intp[::1] label_indices):
        # Determine the number of gradients and hessians to be considered by the upcoming search...
        cdef float64[::1, :] gradients
        cdef intp num_gradients

        if label_indices is None:
            gradients = self.gradients
            num_gradients = gradients.shape[1]
        else:
            num_gradients = label_indices.shape[0]

        # To avoid array-recreation each time the search will be updated, the arrays for storing the sums of gradients
        # and hessians are initialized once at this point. If the arrays from the previous search have the correct size,
        # they are reused.
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians
        cdef intp num_hessians

        if sums_of_gradients is None or sums_of_gradients.shape[0] != num_gradients:
            sums_of_gradients = array_float64(num_gradients)
            self.sums_of_gradients = sums_of_gradients
            num_hessians = __triangular_number(num_gradients)
            sums_of_hessians = array_float64(num_hessians)
            self.sums_of_hessians = sums_of_hessians
        else:
            sums_of_hessians = self.sums_of_hessians

        # Reset the sums of gradients and hessians to 0...
        sums_of_gradients[:] = 0
        sums_of_hessians[:] = 0

        # Reset the accumulated sums of gradients and hessians to None...
        self.accumulated_sums_of_gradients = None
        self.accumulated_sums_of_hessians = None

        # Store the given label indices...
        self.label_indices = label_indices

    cdef void update_search(self, intp example_index, uint32 weight):
        # Class members
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef intp[::1] label_indices = self.label_indices
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp i, c, c2, l, l2, offset

        # Add the gradients and hessians of the example at the given index (weighted by the given weight) to the current
        # sum of gradients and hessians...
        i = 0

        for c in range(num_gradients):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])
            offset = __triangular_number(l)

            for c2 in range(c + 1):
                l2 = offset + get_index(c2, label_indices)
                sums_of_hessians[i] += (weight * hessians[example_index, l2])
                i += 1

    cdef void reset_search(self):
        # Class members
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        # The number of gradients
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # The number of hessians
        cdef intp num_hessians = sums_of_hessians.shape[0]
        # Temporary variables
        cdef intp c
        # Update the arrays that store the accumulated sums of gradients and hessians...
        cdef float64[::1] accumulated_sums_of_gradients = self.accumulated_sums_of_gradients
        cdef float64[::1] accumulated_sums_of_hessians

        if accumulated_sums_of_gradients is None:
            accumulated_sums_of_gradients = array_float64(num_gradients)
            self.accumulated_sums_of_gradients = accumulated_sums_of_gradients
            accumulated_sums_of_hessians = array_float64(num_hessians)
            self.accumulated_sums_of_hessians = accumulated_sums_of_hessians

            for c in range(num_gradients):
                accumulated_sums_of_gradients[c] = sums_of_gradients[c]
                sums_of_gradients[c] = 0

            for c in range(num_hessians):
                accumulated_sums_of_hessians[c] = sums_of_hessians[c]
                sums_of_hessians[c] = 0
        else:
            accumulated_sums_of_hessians = self.accumulated_sums_of_hessians

            for c in range(num_gradients):
                accumulated_sums_of_gradients[c] += sums_of_gradients[c]
                sums_of_gradients[c] = 0

            for c in range(num_hessians):
                accumulated_sums_of_hessians[c] += sums_of_hessians[c]
                sums_of_hessians[c] = 0

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated):
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64[::1] quality_scores = prediction.quality_scores
        cdef float64[::1] sums_of_gradients = self.accumulated_sums_of_gradients if accumulated else self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.accumulated_sums_of_hessians if accumulated else self.sums_of_hessians
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]

        # To avoid array recreation each time this function is called, the arrays for storing predictions and quality
        # scores are only (re-)initialized if they have not been initialized yet, or if they have the wrong size.
        if predicted_scores is None or predicted_scores.shape[0] != num_gradients:
            predicted_scores = array_float64(num_gradients)
            prediction.predicted_scores = predicted_scores
            quality_scores = array_float64(num_gradients)
            prediction.quality_scores = quality_scores

        # The overall quality score, i.e. the sum of the quality scores for each label plus the L2 regularization term
        cdef float64 overall_quality_score = 0
        # Temporary variables
        cdef float64[::1] total_sums_of_gradients, total_sums_of_hessians
        cdef intp[::1] label_indices
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, c2, l, l2

        if uncovered:
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            label_indices = self.label_indices

        # For each label, calculate the score to be predicted, as well as a quality score...
        for c in range(num_gradients):
            sum_of_gradients = sums_of_gradients[c]
            c2 = __triangular_number(c + 1) - 1
            sum_of_hessians = sums_of_hessians[c2]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                l2 = __triangular_number(l + 1) - 1
                sum_of_hessians = total_sums_of_hessians[l2] - sum_of_hessians

            # Calculate score to be predicted for the current label...
            # Note: As the sign of the gradients was inverted in the function `calculate_default_scores`, it must be
            # reverted again in the following.
            score = sum_of_hessians + l2_regularization_weight
            score = sum_of_gradients / score if score != 0 else 0
            predicted_scores[c] = score

            # Calculate the quality score for the current label...
            score_pow = pow(score, 2)
            score = (-sum_of_gradients * score) + (0.5 * score_pow * sum_of_hessians)
            quality_scores[c] = score + (0.5 * l2_regularization_weight * score_pow)
            overall_quality_score += score

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(predicted_scores)
        prediction.overall_quality_score = overall_quality_score

        return prediction

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated):
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef Prediction prediction = <Prediction>self.prediction
        cdef float64[::1] sums_of_gradients = self.accumulated_sums_of_gradients if accumulated else self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.accumulated_sums_of_hessians if accumulated else self.sums_of_hessians
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # Temporary variables
        cdef float64[::1] gradients, hessians, total_sums_of_gradients, total_sums_of_hessians
        cdef intp[::1] label_indices
        cdef intp num_hessians, c, c2, l, l2, i, offset

        if uncovered:
            label_indices = self.label_indices
            num_hessians = sums_of_hessians.shape[0]
            gradients = array_float64(num_gradients)
            hessians = array_float64(num_hessians)
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            i = 0

            for c in range(num_gradients):
                l = get_index(c, label_indices)
                gradients[c] = total_sums_of_gradients[l] - sums_of_gradients[c]
                offset = __triangular_number(l)

                for c2 in range(c + 1):
                    l2 = offset + get_index(c2, label_indices)
                    hessians[i] = total_sums_of_hessians[l2] - sums_of_hessians[i]
                    i += 1
        else:
            gradients = sums_of_gradients
            hessians = sums_of_hessians

        # Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
        cdef float64[::1] scores = __dsysv_float64(hessians, gradients, l2_regularization_weight)
        prediction.predicted_scores = scores

        # Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
        cdef float64 overall_quality_score = -__ddot_float64(scores, gradients)
        cdef float64[::1] tmp = __dspmv_float64(hessians, scores)
        overall_quality_score += 0.5 * __ddot_float64(scores, tmp)

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(scores)
        prediction.overall_quality_score = overall_quality_score

        return prediction

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        # Class members
        cdef float64[::1, :] expected_scores = self.expected_scores
        cdef float64[::1, :] current_scores = self.current_scores
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1, :] hessians = self.hessians
        # The total number of labels
        cdef intp num_labels = gradients.shape[1]
        # The number of predicted labels
        cdef intp num_predicted_labels = predicted_scores.shape[0]
        # An array for caching pre-calculated values
        cdef float64[::1] exponentials = array_float64(num_labels)
        # Temporary variables
        cdef float64 expected_score, exponential, score, sum_of_exponentials, sum_of_exponentials_pow, tmp
        cdef intp c, c2, l, j

        # Traverse the labels for which the new rule predicts to update the currently predicted scores...
        for c in range(num_predicted_labels):
            l = get_index(c, label_indices)
            current_scores[example_index, l] += predicted_scores[c]

        # Traverse the labels of the current example to create arrays of expected scores and exponentials that are
        # shared among the upcoming calculations of gradients and hessians...
        sum_of_exponentials = 1

        for c in range(num_labels):
            expected_score = expected_scores[example_index, c]
            exponential = exp(-expected_score * current_scores[example_index, c])
            exponentials[c] = exponential
            sum_of_exponentials += exponential

        sum_of_exponentials_pow = pow(sum_of_exponentials, 2)

        # Traverse the labels again to update the gradients and hessians...
        j = 0

        for c in range(num_labels):
            expected_score = expected_scores[example_index, c]
            exponential = exponentials[c]
            score = current_scores[example_index, c]

            # Calculate the first derivative (gradient) of the loss function with respect to the current label and add
            # it to the matrix of gradients...
            tmp = gradients[example_index, c]
            tmp = (expected_score * exponential) / sum_of_exponentials
            # Note: The sign of the gradient is inverted (from negative to positive), because otherwise, when using the
            # sums of gradients as the ordinates for solving a system of linear equations in the function
            # `evaluate_label_dependent_predictions`, the sign must be inverted again...
            gradients[example_index, c] = tmp

            # Calculate the second derivatives (hessians) of the loss function with respect to the current label and
            # each of the other labels and add them to the matrix of hessians...
            for c2 in range(c):
                tmp = hessians[example_index, j]
                tmp = exp(-expected_scores[example_index, c2] * current_scores[example_index, c2] - expected_score * score)
                tmp = (expected_scores[example_index, c2] * expected_score * tmp) / sum_of_exponentials_pow
                hessians[example_index, j] = -tmp
                j += 1

            # Calculate the second derivative (hessian) of the loss function with respect to the current label and add
            # it to the matrix of hessians...
            tmp = hessians[example_index, j]
            tmp = (pow(expected_score, 2) * exponential * (sum_of_exponentials - exponential)) / sum_of_exponentials_pow
            hessians[example_index, j] = tmp
            j += 1


cdef inline intp __triangular_number(intp n):
    """
    Computes and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.

    :param n:   A scalar of dtype `intp`, representing the order of the triangular number
    :return:    A scalar of dtype `intp`, representing the n-th triangular number
    """
    return (n * (n + 1)) // 2


cdef inline float64 __ddot_float64(float64[::1] x, float64[::1] y):
    """
    Computes and returns the dot product x * y of two vectors using BLAS' DDOT routine (see
    http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga75066c4825cb6ff1c8ec4403ef8c843a.html).

    :param x:   An array of dtype `float64`, shape (n), representing the first vector x
    :param y:   An array of dtype `float64`, shape (n), representing the second vector y
    :return:    A scalar of dtype `float64`, representing the result of the dot product x * y
    """
    # The number of elements in the arrays x and y
    cdef int n = x.shape[0]
    # Storage spacing between the elements of the arrays x and y
    cdef int inc = 1
    # Invoke the DDOT routine...
    cdef float64 result = ddot(&n, &x[0], &inc, &y[0], &inc)
    return result


cdef inline float64[::1] __dspmv_float64(float64[::1] a, float64[::1] x):
    """
    Computes and returns the solution to the matrix-vector operation A * x using BLAS' DSPMV routine (see
    http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gab746575c4f7dd4eec72e8110d42cefe9.html).
    This function expects A to be a double-precision symmetric matrix with shape `(n, n)` and x a double-precision array
    with shape `(n)`.

    DSPMV expects the matrix A to be supplied in packed form, i.e., as an array with shape `(n * (n + 1) // 2 )` that
    consists of the columns of A appended to each other and omitting all unspecified elements.

    :param a:   An array of dtype `float64`, shape `(n * (n + 1) // 2)`, representing the elements in the upper-right
                triangle of the matrix A in a packed form
    :param x:   An array of dtype `float64`, shape `(n)`, representing the elements in the array x
    :return:    An array of dtype `float64`, shape `(n)`, representing the result of the matrix-vector operation A * x
    """
    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # The number of rows and columns of the matrix A
    cdef int n = x.shape[0]
    # A scalar to be multiplied with the matrix A
    cdef float64 alpha = 1
    # The increment for the elements of x
    cdef int incx = 1
    # A scalar to be multiplied with vector y
    cdef float64 beta = 0
    # An array of dtype `float64`, shape `(n)`. Will contain the result of A * x
    cdef float64[::1] y = array_float64(n)
    # The increment for the elements of y
    cdef int incy = 1
    # Invoke the DSPMV routine...
    dspmv(uplo, &n, &alpha, &a[0], &x[0], &incx, &beta, &y[0], &incy)
    return y


cdef inline float64[::1] __dsysv_float64(float64[::1] coefficients, float64[::1] ordinates,
                                         float64 l2_regularization_weight):
    """
    Computes and returns the solution to a system of linear equations A * X = B using LAPACK's DSYSV solver (see
    http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html).
    DSYSV requires A to be a double-precision matrix with shape `(num_equations, num_equations)`, representing the
    coefficients, and B to be a double-precision matrix with shape `(num_equations, nrhs)`, representing the ordinates.
    X is a matrix of unknowns with shape `(num_equations, nrhs)`.

    DSYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to the system
    of linear equations. To retain their state, this function will copy the given arrays before invoking DSYSV.

    Furthermore, DSYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the upper-right
    triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency, this function expects
    the coefficients to be given as an array with shape `num_equations * (num_equations + 1) // 2`, representing the
    elements of the upper-right triangle of A, where the columns are appended to each other and unspecified elements are
    omitted. This function will implicitly convert the given array into a matrix that is suited for DSYSV.

    Optionally, this function allows to specify a weight to be used for L2 regularization. The given weight is added to
    each element on the diagonal of the matrix of coefficients A.

    :param coefficients:                An array of dtype `float64`, shape `num_equations * (num_equations + 1) // 2)`,
                                        representing coefficients
    :param ordinates:                   An array of dtype `float64`, shape `(num_equations)`, representing the ordinates
    :param l2_regularization_weight:    A scalar of dtype `float64`, representing the weight of the L2 regularization
    :return:                            An array of dtype `float64`, shape `(num_equations)`, representing the solution
                                        to the system of linear equations
    """
    cdef float64[::1] result
    cdef float64 tmp
    cdef intp r, c, i
    # The number of linear equations
    cdef int n = ordinates.shape[0]
    # Create the array A by copying the array `coefficients`. DSYSV requires the array A to be Fortran-contiguous...
    cdef float64[::1, :] a = fortran_matrix_float64(n, n)
    i = 0

    for c in range(n):
        for r in range(c + 1):
            tmp = coefficients[i]

            if r == c:
                tmp += l2_regularization_weight

            a[r, c] = tmp
            i += 1

    # Create the array B by copying the array `ordinates`. It will be overwritten with the solution to the system of
    # linear equations. DSYSV requires the array B to be Fortran-contiguous...
    cdef float64[::1, :] b = fortran_matrix_float64(n, 1)

    for r in range(n):
        b[r, 0] = ordinates[r]

    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # The number of right-hand sides, i.e, the number of columns of the matrix B
    cdef int nrhs = b.shape[1]
    # Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    cdef int info
    # We must query optimal value for the argument `lwork` (the length of the working array `work`)...
    cdef double worksize
    cdef int lwork = -1  # -1 means that the optimal value should be queried
    dsysv(uplo, &n, &nrhs, &a[0, 0], &n, <int*>0, &b[0, 0], &n, &worksize, &lwork, &info)  # Queries the optimal value
    lwork = <int>worksize
    # Allocate the working array...
    cdef double* work = <double*>malloc(lwork * sizeof(double))
    # Allocate another working array...
    cdef int* ipiv = <int*>malloc(n * sizeof(int))

    try:
        # Run the DSYSV solver...
        dsysv(uplo, &n, &nrhs, &a[0, 0], &n, ipiv, &b[0, 0], &n, work, &lwork, &info)

        if info == 0:
            # The solution has been computed successfully...
            result = b[:, 0]
            return result
        else:
            # An error occurred...
            raise ArithmeticError('DSYSV terminated with non-zero info code: ' + str(info))
    finally:
        # Free the allocated memory...
        free(<void*>ipiv)
        free(<void*>work)
