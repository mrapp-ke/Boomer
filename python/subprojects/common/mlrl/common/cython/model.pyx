"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.algorithm cimport copy
from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from abc import abstractmethod

import numpy as np

SERIALIZATION_VERSION = 1


cdef class Body:
    """
    The body of a rule.
    """
    pass


cdef class EmptyBody(Body):
    """
    A body that does not contain any conditions.
    """
    pass


cdef class ConjunctiveBody(Body):
    """
    A body that is given as a conjunction of several conditions.
    """

    def __cinit__(self, const uint32[::1] leq_indices, const float32[::1] leq_thresholds, const uint32[::1] gr_indices,
                  const float32[::1] gr_thresholds, const uint32[::1] eq_indices, const float32[::1] eq_thresholds,
                  const uint32[::1] neq_indices, const float32[::1] neq_thresholds):
        """
        :param leq_indices:     A contiguous array of type `uint32`, shape `(num_leq_conditions)`, that stores the
                                feature indices of the conditions that use the <= operator
        :param leq_thresholds:  A contiguous array of type `float32`, shape `(num_leq_conditions)` that stores the
                                thresholds of the conditions that use the <= operator
        :param gr_indices:      A contiguous array of type `uint32`, shape `(num_gr_conditions)`, that stores the
                                feature indices of the conditions that use the > operator
        :param gr_thresholds:   A contiguous array of type `float32`, shape `(num_gr_conditions)` that stores the
                                thresholds of the conditions that use the > operator
        :param eq_indices:      A contiguous array of type `uint32`, shape `(num_eq_conditions)`, that stores the
                                feature indices of the conditions that use the == operator
        :param eq_thresholds:   A contiguous array of type `float32`, shape `(num_eq_conditions)` that stores the
                                thresholds of the conditions that use the == operator
        :param neq_indices:     A contiguous array of type `uint32`, shape `(num_neq_conditions)`, that stores the
                                feature indices of the conditions that use the != operator
        :param neq_thresholds:  A contiguous array of type `float32`, shape `(num_neq_conditions)` that stores the
                                thresholds of the conditions that use the != operator
        """
        self.leq_indices = np.asarray(leq_indices) if leq_indices is not None else None
        self.leq_thresholds = np.asarray(leq_thresholds) if leq_thresholds is not None else None
        self.gr_indices = np.asarray(gr_indices) if gr_indices is not None else None
        self.gr_thresholds = np.asarray(gr_thresholds) if gr_thresholds is not None else None
        self.eq_indices = np.asarray(eq_indices) if eq_indices is not None else None
        self.eq_thresholds = np.asarray(eq_thresholds) if eq_thresholds is not None else None
        self.neq_indices = np.asarray(neq_indices) if neq_indices is not None else None
        self.neq_thresholds = np.asarray(neq_thresholds) if neq_thresholds is not None else None


cdef class Head:
    """
    The head of a rule.
    """
    pass


cdef class CompleteHead(Head):
    """
    A head that predicts for all available labels.
    """

    def __cinit__(self, const float64[::1] scores):
        """
        :param scores: A contiguous array of type `float64`, shape `(num_predictions)` that stores the predicted scores
        """
        self.scores = np.asarray(scores)


cdef class PartialHead(Head):
    """
    A head that predicts for a subset of the available labels.
    """

    def __cinit__(self, const uint32[::1] indices, const float64[::1] scores):
        """
        :param indices: A contiguous array of type `uint32`, shape `(num_predictions)` that stores the label indices
        :param scores:  A contiguous array of type `float64`, shape `(num_predictions)` that stores the predicted scores
        """
        self.indices = np.asarray(indices)
        self.scores = np.asarray(scores)


class RuleModelVisitor:
    """
    Defines the methods that must be implemented by a visitor that accesses the bodies and heads of the rules in a
    `RuleModel` according to the visitor pattern.
    """

    @abstractmethod
    def visit_empty_body(self, body: EmptyBody):
        """
        Must be implemented by subclasses in order to visit bodies of rules that do not contain any conditions.

        :param body: An `EmptyBody` to be visited
        """
        pass

    @abstractmethod
    def visit_conjunctive_body(self, body: ConjunctiveBody):
        """
        Must be implemented by subclasses in order to visit the bodies of rule that are given as a conjunction of
        several conditions.

        :param body: A `ConjunctiveBody` to be visited
        """
        pass

    @abstractmethod
    def visit_complete_head(self, head: CompleteHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict for all available labels.

        :param head: A `CompleteHead` to be visited
        """
        pass

    @abstractmethod
    def visit_partial_head(self, head: PartialHead):
        """
        Must be implemented by subclasses in order to visit the heads of rules that predict for a subset of the
        available labels.

        :param head: A `PartialHead` to be visited
        """
        pass


cdef class RuleModel:
    """
    A wrapper for the C++ class `RuleModel`.
    """

    def get_num_rules(self) -> int:
        """
        Returns the total number of rules in the model.

        :return The total number of rules in the model
        """
        return self.model_ptr.get().getNumRules()

    def get_num_used_rules(self) -> int:
        """
        Returns the number of used rules in the model.

        :return The number of used rules in the model
        """
        return self.model_ptr.get().getNumUsedRules()

    def set_num_used_rules(self, int num_used_rules):
        """
        Sets the number of used rules in the model.

        :param num_used_rules: The number of used rules to be set
        """
        self.model_ptr.get().setNumUsedRules(num_used_rules)

    def visit(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of the rules in the model.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """
        cdef RuleModelVisitorWrapper wrapper = RuleModelVisitorWrapper.__new__(RuleModelVisitorWrapper, visitor)
        wrapper.visit(self)

    def __getstate__(self):
        cdef RuleModelSerializer serializer = RuleModelSerializer.__new__(RuleModelSerializer)
        cdef object state = serializer.serialize(self)
        return state

    def __setstate__(self, state):
        cdef RuleModelSerializer serializer = RuleModelSerializer.__new__(RuleModelSerializer)
        serializer.deserialize(self, state)


cdef class ModelBuilder:
    """
    A wrapper for the pure virtual C++ class `IModelBuilder`.
    """
    pass


cdef unique_ptr[IBody] __create_body(object state):
    cdef const float32[::1] leq_thresholds
    cdef const uint32[::1] leq_indices
    cdef const float32[::1] gr_thresholds
    cdef const uint32[::1] gr_indices
    cdef const float32[::1] eq_thresholds
    cdef const uint32[::1] eq_indices
    cdef const float32[::1] neq_thresholds
    cdef const uint32[::1] neq_indices

    if state is None:
        return <unique_ptr[IBody]>make_unique[EmptyBodyImpl]()
    else:
        leq_thresholds = state[0]
        leq_indices = state[1]
        gr_thresholds = state[2]
        gr_indices = state[3]
        eq_thresholds = state[4]
        eq_indices = state[5]
        neq_thresholds = state[6]
        neq_indices = state[7]
        return __create_conjunctive_body(leq_thresholds, leq_indices, gr_thresholds, gr_indices, eq_thresholds,
                                         eq_indices, neq_thresholds, neq_indices)


cdef unique_ptr[IBody] __create_conjunctive_body(const float32[::1] leq_thresholds, const uint32[::1] leq_indices,
                                                 const float32[::1] gr_thresholds, const uint32[::1] gr_indices,
                                                 const float32[::1] eq_thresholds, const uint32[::1] eq_indices,
                                                 const float32[::1] neq_thresholds, const uint32[::1] neq_indices):
    cdef uint32 num_leq = leq_thresholds.shape[0]
    cdef uint32 num_gr = gr_thresholds.shape[0]
    cdef uint32 num_eq = eq_thresholds.shape[0]
    cdef uint32 num_neq = neq_thresholds.shape[0]
    cdef unique_ptr[ConjunctiveBodyImpl] body_ptr = make_unique[ConjunctiveBodyImpl](num_leq, num_gr, num_eq, num_neq)

    cdef const float32* thresholds_begin = &leq_thresholds[0]
    cdef const float32* thresholds_end = &leq_thresholds[num_leq]
    cdef ConjunctiveBodyImpl.threshold_iterator threshold_iterator = body_ptr.get().leq_thresholds_begin()
    copy(thresholds_begin, thresholds_end, threshold_iterator)
    cdef const uint32* indices_begin = &leq_indices[0]
    cdef const uint32* indices_end = &leq_indices[num_leq]
    cdef ConjunctiveBodyImpl.index_iterator index_iterator = body_ptr.get().leq_indices_begin()
    copy(indices_begin, indices_end, index_iterator)

    thresholds_begin = &gr_thresholds[0]
    thresholds_end = &gr_thresholds[num_gr]
    threshold_iterator = body_ptr.get().gr_thresholds_begin()
    copy(thresholds_begin, thresholds_end, threshold_iterator)
    indices_begin = &gr_indices[0]
    indices_end = &gr_indices[num_gr]
    index_iterator = body_ptr.get().gr_indices_begin()
    copy(indices_begin, indices_end, index_iterator)

    thresholds_begin = &eq_thresholds[0]
    thresholds_end = &eq_thresholds[num_eq]
    threshold_iterator = body_ptr.get().eq_thresholds_begin()
    copy(thresholds_begin, thresholds_end, threshold_iterator)
    indices_begin = &eq_indices[0]
    indices_end = &eq_indices[num_eq]
    index_iterator = body_ptr.get().eq_indices_begin()
    copy(indices_begin, indices_end, index_iterator)

    thresholds_begin = &neq_thresholds[0]
    thresholds_end = &neq_thresholds[num_neq]
    threshold_iterator = body_ptr.get().neq_thresholds_begin()
    copy(thresholds_begin, thresholds_end, threshold_iterator)
    indices_begin = &neq_indices[0]
    indices_end = &neq_indices[num_neq]
    index_iterator = body_ptr.get().neq_indices_begin()
    copy(indices_begin, indices_end, index_iterator)

    return <unique_ptr[IBody]>move(body_ptr)


cdef unique_ptr[IHead] __create_head(object state):
    cdef const float64[::1] scores = state[0]
    cdef const uint32[::1] indices

    if len(state) > 1:
        indices = state[1]
        return __create_partial_head(scores, indices)
    else:
        return __create_complete_head(scores)


cdef unique_ptr[IHead] __create_complete_head(const float64[::1] scores):
    cdef uint32 num_elements = scores.shape[0]
    cdef unique_ptr[CompleteHeadImpl] head_ptr = make_unique[CompleteHeadImpl](num_elements)
    cdef const float64* scores_begin = &scores[0]
    cdef const float64* scores_end = &scores[num_elements]
    cdef CompleteHeadImpl.score_iterator score_iterator = head_ptr.get().scores_begin()
    copy(scores_begin, scores_end, score_iterator)
    return <unique_ptr[IHead]>move(head_ptr)


cdef unique_ptr[IHead] __create_partial_head(const float64[::1] scores, const uint32[::1] indices):
    cdef uint32 num_elements = scores.shape[0]
    cdef unique_ptr[PartialHeadImpl] head_ptr = make_unique[PartialHeadImpl](num_elements)
    cdef const float64* scores_begin = &scores[0]
    cdef const float64* scores_end = &scores[num_elements]
    cdef PartialHeadImpl.score_iterator score_iterator = head_ptr.get().scores_begin()
    copy(scores_begin, scores_end, score_iterator)
    cdef const uint32* indices_begin = &indices[0]
    cdef const uint32* indices_end = &indices[num_elements]
    cdef PartialHeadImpl.index_iterator index_iterator = head_ptr.get().indices_begin()
    copy(indices_begin, indices_end, index_iterator)
    return <unique_ptr[IHead]>move(head_ptr)


cdef class RuleModelSerializer:
    """
    Allows to serialize and deserialize the rules that are contained by a `RuleModel`.
    """

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        body_state = None
        rule_state = [body_state, None]
        self.state.append(rule_state)

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_leq = body.getNumLeq()
        cdef uint32 num_gr = body.getNumGr()
        cdef uint32 num_eq = body.getNumEq()
        cdef uint32 num_neq = body.getNumNeq()
        body_state = (np.asarray(<float32[:num_leq]>body.leq_thresholds_cbegin()) if num_leq > 0 else None,
                      np.asarray(<uint32[:num_leq]>body.leq_indices_cbegin()) if num_leq > 0 else None,
                      np.asarray(<float32[:num_gr]>body.gr_thresholds_cbegin()) if num_gr > 0 else None,
                      np.asarray(<uint32[:num_gr]>body.gr_indices_cbegin()) if num_gr > 0 else None,
                      np.asarray(<float32[:num_eq]>body.eq_thresholds_cbegin()) if num_eq > 0 else None,
                      np.asarray(<uint32[:num_eq]>body.eq_indices_cbegin()) if num_eq > 0 else None,
                      np.asarray(<float32[:num_neq]>body.neq_thresholds_cbegin()) if num_neq > 0 else None,
                      np.asarray(<uint32[:num_neq]>body.neq_indices_cbegin()) if num_neq > 0 else None)
        rule_state = [body_state, None]
        self.state.append(rule_state)

    cdef __visit_complete_head(self, const CompleteHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        rule_state = self.state[len(self.state) - 1]
        head_state = (np.asarray(<float64[:num_elements]>head.scores_cbegin()),)
        rule_state[1] = head_state

    cdef __visit_partial_head(self, const PartialHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        rule_state = self.state[len(self.state) - 1]
        head_state = (np.asarray(<float64[:num_elements]>head.scores_cbegin()),
                      np.asarray(<uint32[:num_elements]>head.indices_cbegin()))
        rule_state[1] = head_state

    def serialize(self, RuleModel model) -> object:
        """
        Creates and returns a state, which may be serialized using Python's pickle mechanism, from the rules that are
        contained by a given `RuleModel`.

        :param model:   The model that contains the rules to be serialized
        :return:        The state that has been created
        """
        self.state = []
        model.model_ptr.get().visit(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__visit_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))
        cdef uint32 num_used_rules = model.model_ptr.get().getNumUsedRules()
        return (SERIALIZATION_VERSION, (self.state, num_used_rules))

    def deserialize(self, RuleModel model, object state):
        """
        Deserializes the rules that are contained by a given state and adds them to a `RuleModel`.

        :param model:   The model, the deserialized rules should be added to
        :param state:   A state that has previously been created via the function `serialize`
        """
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError(
                'Version of the serialized model is ' + str(version) + ', expected ' + str(SERIALIZATION_VERSION))

        model_state = state[1]
        cdef list rule_list = model_state[0]
        cdef uint32 num_rules = len(rule_list)
        cdef unique_ptr[RuleModelImpl] rule_model_ptr = make_unique[RuleModelImpl]()
        cdef object rule_state
        cdef uint32 i

        for i in range(num_rules):
            rule_state = rule_list[i]
            rule_model_ptr.get().addRule(move(__create_body(rule_state[0])), move(__create_head(rule_state[1])))

        cdef uint32 num_used_rules = model_state[1]
        rule_model_ptr.get().setNumUsedRules(num_used_rules)
        model.model_ptr = move(rule_model_ptr)


cdef class RuleModelVisitorWrapper:
    """
    Wraps a `RuleModelVisitor` and invokes its methods when visiting the bodies and heads of a `RuleModel`.
    """

    def __cinit__(self, object visitor):
        """
        :param visitor: The `RuleModelVisitor` to be wrapped
        """
        self.visitor = visitor

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        self.visitor.visit_empty_body(EmptyBody.__new__(EmptyBody))

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_leq = body.getNumLeq()
        cdef const uint32[::1] leq_indices = <uint32[:num_leq]>body.leq_indices_cbegin() if num_leq > 0 else None
        cdef const float32[::1] leq_thresholds = <float32[:num_leq]>body.leq_thresholds_cbegin() if num_leq > 0 else None
        cdef uint32 num_gr = body.getNumGr()
        cdef const uint32[::1] gr_indices = <uint32[:num_gr]>body.gr_indices_cbegin() if num_gr > 0 else None
        cdef const float32[::1] gr_thresholds = <float32[:num_gr]>body.gr_thresholds_cbegin() if num_gr > 0 else None
        cdef uint32 num_eq = body.getNumEq()
        cdef const uint32[::1] eq_indices = <uint32[:num_eq]>body.eq_indices_cbegin() if num_eq > 0 else None
        cdef const float32[::1] eq_thresholds = <float32[:num_eq]>body.eq_thresholds_cbegin() if num_eq > 0 else None
        cdef uint32 num_neq = body.getNumNeq()
        cdef const uint32[::1] neq_indices = <uint32[:num_neq]>body.neq_indices_cbegin() if num_neq > 0 else None
        cdef const float32[::1] neq_thresholds = <float32[:num_neq]>body.neq_thresholds_cbegin() if num_neq > 0 else None
        self.visitor.visit_conjunctive_body(ConjunctiveBody.__new__(ConjunctiveBody, leq_indices, leq_thresholds,
                                                                    gr_indices, gr_thresholds, eq_indices,
                                                                    eq_thresholds, neq_indices, neq_thresholds))

    cdef __visit_complete_head(self, const CompleteHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef const float64[::1] scores = <float64[:num_elements]>head.scores_cbegin()
        self.visitor.visit_complete_head(CompleteHead.__new__(CompleteHead, scores))

    cdef __visit_partial_head(self, const PartialHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef const uint32[::1] indices = <uint32[:num_elements]>head.indices_cbegin()
        cdef const float64[::1] scores = <float64[:num_elements]>head.scores_cbegin()
        self.visitor.visit_partial_head(PartialHead.__new__(PartialHead, indices, scores))

    cdef visit(self, RuleModel model):
        """
        Visits a specific model.

        :param model: The `RuleModel` to be visited
        """
        model.model_ptr.get().visitUsed(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__visit_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))
