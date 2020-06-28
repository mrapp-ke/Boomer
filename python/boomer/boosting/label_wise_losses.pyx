"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement loss functions that are applied example- and label-wise.
"""
from boomer.common._arrays cimport array_float64, fortran_matrix_float64, get_index
from boomer.boosting.differentiable_losses cimport _convert_label_into_score, _l2_norm_pow

from libc.math cimport pow, exp


cdef class LabelWiseDifferentiableLoss(DecomposableDifferentiableLoss):
    """
    A base class for all differentiable loss functions that are applied label-wise.
    """

    cdef float64 _gradient(self, float64 expected_score, float64 current_score):
        """
        Must be implemented by subclasses to calculate the gradient (first derivative of the loss function) for a 
        certain example and label.
        
        :param expected_score:  A scalar of dtype float64, representing the expected score for the respective example 
                                and label
        :param current_score:   A scalar of dtype float64, representing the currently predicted score for the respective
                                example and label
        :return:                A scalar of dtype float64, representing the gradient that has been calculated
        """
        pass

    cdef float64 _hessian(self, float64 expected_score, float64 current_score):
        """
        Must be implemented by subclasses to calculate the hessian (second derivative of the loss function) for a 
        certain example and label.
        
        :param expected_score:  A scalar of dtype float64, representing the expected score for the respective example 
                                and label
        :param current_score:   A scalar of dtype float64, representing the currently predicted score for the respective
                                example and label
        :return:                A scalar of dtype float64, representing the hessian that has been calculated
        """
        pass

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the optimal
                                         scores to be predicted by rules. Increasing this value causes the model to be
                                         more conservative, setting it to 0 turns off L2 regularization entirely
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
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[::1, :] current_scores = fortran_matrix_float64(num_examples, num_labels)
        # A matrix that stores the gradients for each example and label
        cdef float64[::1, :] gradients = fortran_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        # A matrix that stores the hessians for each example and label
        cdef float64[::1, :] hessians = fortran_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of hessians
        cdef float64[::1] total_sums_of_hessians = array_float64(num_labels)
        # An array that stores the scores that are predicted by the default rule
        cdef float64[::1] scores = array_float64(num_labels)
        # Temporary variables
        cdef float64 sum_of_gradients, sum_of_hessians, expected_score, score, tmp
        cdef intp c, r

        for c in range(num_labels):
            # Column-wise sum up the gradients and hessians for the current label...
            sum_of_gradients = 0
            sum_of_hessians = 0

            for r in range(num_examples):
                # Convert ground truth label into expected score...
                expected_score = _convert_label_into_score(y[r, c])
                expected_scores[r, c] = expected_score

                # Calculate gradient for the current example and label...
                tmp = self._gradient(expected_score, 0)
                sum_of_gradients += tmp

                # Calculate hessian for the current example and label...
                tmp = self._hessian(expected_score, 0)
                sum_of_hessians += tmp

            # Calculate optimal score to be predicted by the default rule for the current label...
            score = -sum_of_gradients / (sum_of_hessians + l2_regularization_weight)
            scores[c] = score

            # Traverse column again to calculate updated gradients based on the calculated score...
            for r in range(num_examples):
                expected_score = expected_scores[r, c]

                # Calculate updated gradient for the current example and label...
                tmp = self._gradient(expected_score, score)
                gradients[r, c] = tmp

                # Calculate updated gradient for the current example and label...
                tmp = self._hessian(expected_score, score)
                hessians[r, c] = tmp

                # Store the score that is currently predicted for the current example and label...
                current_scores[r, c] = score

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
        # The number of labels
        cdef intp num_labels = total_sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c

        # Reset total sums of gradients and hessians to 0...
        for c in range(num_labels):
            total_sums_of_gradients[c] = 0
            total_sums_of_hessians[c] = 0

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        # Class members
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The number of labels
        cdef intp num_labels = total_sums_of_gradients.shape[0]
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # Temporary variables
        cdef intp c

        # For each label, add the gradient and hessian of the example at the given index (weighted by the given weight)
        # to the total sums of gradients and hessians...
        for c in range(num_labels):
            total_sums_of_gradients[c] += (signed_weight * gradients[example_index, c])
            total_sums_of_hessians[c] += (signed_weight * hessians[example_index, c])

    cdef void begin_search(self, intp[::1] label_indices):
        # Determine the number of labels to be considered by the upcoming search...
        cdef float64[::1] total_sums_of_gradients
        cdef intp num_labels, c

        if label_indices is None:
            total_sums_of_gradients = self.total_sums_of_gradients
            num_labels = total_sums_of_gradients.shape[0]
        else:
            num_labels = label_indices.shape[0]

        # To avoid array-recreation each time the search will be updated, the arrays for storing the sums of gradients
        # and hessians, as well as the arrays for storing predictions and quality scores, are initialized once at this
        # point. If the arrays from the previous search have the correct size, they are reused.
        cdef LabelIndependentPrediction prediction = self.prediction
        cdef float64[::1] predicted_scores
        cdef float64[::1] quality_scores
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians

        if sums_of_gradients is None or sums_of_gradients.shape[0] != num_labels:
            sums_of_gradients = array_float64(num_labels)
            self.sums_of_gradients = sums_of_gradients
            sums_of_hessians = array_float64(num_labels)
            self.sums_of_hessians = sums_of_hessians
            predicted_scores = array_float64(num_labels)
            prediction.predicted_scores = predicted_scores
            quality_scores = array_float64(num_labels)
            prediction.quality_scores = quality_scores
        else:
            sums_of_hessians = self.sums_of_hessians
            predicted_scores = prediction.predicted_scores
            quality_scores = prediction.quality_scores

        # Reset the sums of gradients and hessians to 0...
        for c in range(num_labels):
            sums_of_gradients[c] = 0
            sums_of_hessians[c] = 0

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
        # The number of labels considered by the current search
        cdef intp num_labels = sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c, l

        # For each label, add the gradient and hessian of the example at the given index (weighted by the given weight)
        # to the current sum of gradients and hessians...
        for c in range(num_labels):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])
            sums_of_hessians[c] += (weight * hessians[example_index, l])

    cdef void reset_search(self):
        # Class members
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        # The number of labels
        cdef intp num_labels = sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp c

        # Update the arrays that store the accumulated sums of gradients and hessians...
        cdef float64[::1] accumulated_sums_of_gradients = self.accumulated_sums_of_gradients
        cdef float64[::1] accumulated_sums_of_hessians

        if accumulated_sums_of_gradients is None:
            accumulated_sums_of_gradients = array_float64(num_labels)
            self.accumulated_sums_of_gradients = accumulated_sums_of_gradients
            accumulated_sums_of_hessians = array_float64(num_labels)
            self.accumulated_sums_of_hessians = accumulated_sums_of_hessians

            for c in range(num_labels):
                accumulated_sums_of_gradients[c] = sums_of_gradients[c]
                sums_of_gradients[c] = 0
                accumulated_sums_of_hessians[c] = sums_of_hessians[c]
                sums_of_hessians[c] = 0
        else:
            accumulated_sums_of_hessians = self.accumulated_sums_of_hessians

            for c in range(num_labels):
                accumulated_sums_of_gradients[c] += sums_of_gradients[c]
                sums_of_gradients[c] = 0
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
        # The number of labels considered by the current search
        cdef intp num_labels = sums_of_gradients.shape[0]
        # The overall quality score, i.e., the sum of the quality scores for each label plus the L2 regularization term
        cdef float64 overall_quality_score = 0
        # Temporary variables
        cdef float64[::1] total_sums_of_gradients, total_sums_of_hessians
        cdef intp[::1] label_indices
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, l

        if uncovered:
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            label_indices = self.label_indices

        # For each label, calculate the score to be predicted, as well as a quality score...
        for c in range(num_labels):
            sum_of_gradients = sums_of_gradients[c]
            sum_of_hessians = sums_of_hessians[c]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                sum_of_hessians = total_sums_of_hessians[l] - sum_of_hessians

            # Calculate score to be predicted for the current label...
            score = sum_of_hessians + l2_regularization_weight
            score = -sum_of_gradients / score if score != 0 else 0
            predicted_scores[c] = score

            # Calculate the quality score for the current label...
            score_pow = pow(score, 2)
            score = (sum_of_gradients * score) + (0.5 * score_pow * sum_of_hessians)
            quality_scores[c] = score + (0.5 * l2_regularization_weight * score_pow)
            overall_quality_score += score

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(predicted_scores)
        prediction.overall_quality_score = overall_quality_score

        return prediction

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        # Class members
        cdef float64[::1, :] gradients = self.gradients
        cdef float64[::1, :] hessians = self.hessians
        cdef float64[::1, :] expected_scores = self.expected_scores
        cdef float64[::1, :] current_scores = self.current_scores
        # The number of predicted labels
        cdef intp num_labels = predicted_scores.shape[0]
        # Temporary variables
        cdef float64 predicted_score, expected_score, current_score
        cdef intp c, l

        # Only the labels that are predicted by the new rule must be considered...
        for c in range(num_labels):
            l = get_index(c, label_indices)
            predicted_score = predicted_scores[c]

            # Retrieve the expected score for the current example and label...
            expected_score = expected_scores[example_index, l]

            # Update the score that is currently predicted for the current example and label...
            current_score = current_scores[example_index, l] + predicted_score
            current_scores[example_index, l] = current_score

            # Update the gradient for the current example and label...
            gradients[example_index, l] = self._gradient(expected_score, current_score)

            # Update the hessian for the current example and label...
            hessians[example_index, l] = self._hessian(expected_score, current_score)


cdef class LabelWiseSquaredErrorLoss(LabelWiseDifferentiableLoss):
    """
    A multi-label variant of the squared error loss that is applied label-wise.
    """

    cdef float64 _gradient(self, float64 expected_score, float64 current_score):
        return 2 * current_score - 2 * expected_score

    cdef float64 _hessian(self, float64 expected_score, float64 current_score):
        return 2


cdef class LabelWiseLogisticLoss(LabelWiseDifferentiableLoss):
    """
    A multi-label variant of the logistic loss that is applied label-wise.
    """

    cdef float64 _gradient(self, float64 expected_score, float64 current_score):
        return -expected_score / (1 + exp(expected_score * current_score))

    cdef float64 _hessian(self, float64 expected_score, float64 current_score):
        cdef float64 exponential = exp(expected_score * current_score)
        return (pow(expected_score, 2) * exponential) / pow(1 + exponential, 2)
