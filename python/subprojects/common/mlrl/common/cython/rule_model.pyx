"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from abc import abstractmethod

import numpy as np

SERIALIZATION_VERSION = 3


cdef class EmptyBody:
    """
    A body of a rule that does not contain any conditions.
    """
    pass


cdef class ConjunctiveBody:
    """
    A body of a rule that is given as a conjunction of several conditions.
    """

    def __cinit__(self, const uint32[::1] leq_indices, const float32[::1] leq_thresholds, const uint32[::1] gr_indices,
                  const float32[::1] gr_thresholds, const uint32[::1] eq_indices, const float32[::1] eq_thresholds,
                  const uint32[::1] neq_indices, const float32[::1] neq_thresholds):
        """
        :param leq_indices:     A contiguous array of type `uint32`, shape `(num_leq_conditions)`, that stores the
                                feature indices of the conditions that use the <= operator or None, if no such
                                conditions are available
        :param leq_thresholds:  A contiguous array of type `float32`, shape `(num_leq_conditions)` that stores the
                                thresholds of the conditions that use the <= operator or None, if no such conditions are
                                available
        :param gr_indices:      A contiguous array of type `uint32`, shape `(num_gr_conditions)`, that stores the
                                feature indices of the conditions that use the > operator or None, if no such conditions
                                are available
        :param gr_thresholds:   A contiguous array of type `float32`, shape `(num_gr_conditions)` that stores the
                                thresholds of the conditions that use the > operator or None, if no such conditions are
                                available
        :param eq_indices:      A contiguous array of type `uint32`, shape `(num_eq_conditions)`, that stores the
                                feature indices of the conditions that use the == operator or None, if no such
                                conditions are available
        :param eq_thresholds:   A contiguous array of type `float32`, shape `(num_eq_conditions)` that stores the
                                thresholds of the conditions that use the == operator or None, if no such conditions are
                                available
        :param neq_indices:     A contiguous array of type `uint32`, shape `(num_neq_conditions)`, that stores the
                                feature indices of the conditions that use the != operator or None, if no such
                                conditions are available
        :param neq_thresholds:  A contiguous array of type `float32`, shape `(num_neq_conditions)` that stores the
                                thresholds of the conditions that use the != operator or None, if no such conditions are
                                available
        """
        self.leq_indices = np.asarray(leq_indices) if leq_indices is not None else None
        self.leq_thresholds = np.asarray(leq_thresholds) if leq_thresholds is not None else None
        self.gr_indices = np.asarray(gr_indices) if gr_indices is not None else None
        self.gr_thresholds = np.asarray(gr_thresholds) if gr_thresholds is not None else None
        self.eq_indices = np.asarray(eq_indices) if eq_indices is not None else None
        self.eq_thresholds = np.asarray(eq_thresholds) if eq_thresholds is not None else None
        self.neq_indices = np.asarray(neq_indices) if neq_indices is not None else None
        self.neq_thresholds = np.asarray(neq_thresholds) if neq_thresholds is not None else None


cdef class CompleteHead:
    """
    A head of a rule that predicts for all available labels.
    """

    def __cinit__(self, const float64[::1] scores not None):
        """
        :param scores: A contiguous array of type `float64`, shape `(num_predictions)` that stores the predicted scores
        """
        self.scores = np.asarray(scores)


cdef class PartialHead:
    """
    A head of a rule that predicts for a subset of the available labels.
    """

    def __cinit__(self, const uint32[::1] indices not None, const float64[::1] scores not None):
        """
        :param indices: A contiguous array of type `uint32`, shape `(num_predictions)` that stores the label indices
        :param scores:  A contiguous array of type `float64`, shape `(num_predictions)` that stores the predicted scores
        """
        self.indices = np.asarray(indices)
        self.scores = np.asarray(scores)


class RuleModelVisitor:
    """
    Defines the methods that must be implemented by a visitor that accesses the bodies and heads of the rules in a
    rule-based model according to the visitor pattern.
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
    A rule-based model.
    """

    cdef IRuleModel* get_rule_model_ptr(self):
        pass

    def get_num_rules(self) -> int:
        """
        Returns the total number of rules in the model, including the default rule, if available.

        :return The total number of rules in the model
        """
        return self.get_rule_model_ptr().getNumRules()

    def get_num_used_rules(self) -> int:
        """
        Returns the number of used rules in the model, including the default rule, if available.

        :return The number of used rules in the model
        """
        return self.get_rule_model_ptr().getNumUsedRules()

    def set_num_used_rules(self, num_used_rules: int):
        """
        Sets the number of used rules in the model, including the default rule, if available.

        :param num_used_rules: The number of used rules to be set
        """
        self.get_rule_model_ptr().setNumUsedRules(num_used_rules)

    def visit(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of all rules that are contained in this model, including the default rule, if
        available.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """
        pass

    def visit_used(self, visitor: RuleModelVisitor):
        """
        Visits the bodies and heads of all used rules that are contained in this model, including the default rule, if
        available.

        :param visitor: The `RuleModelVisitor` that should be used to access the bodies and heads
        """
        pass


cdef class RuleList(RuleModel):
    """
    A rule-based model that stores several rules in an ordered list.
    """

    def __cinit__(self):
        self.visitor = None
        self.state = None

    cdef IRuleModel* get_rule_model_ptr(self):
        return self.rule_list_ptr.get()

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        self.visitor.visit_empty_body(EmptyBody.__new__(EmptyBody))

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_leq = body.getNumLeq()
        cdef const uint32[::1] leq_indices = <uint32[:num_leq]>body.leq_indices_cbegin() if num_leq > 0 else None
        cdef const float32[::1] leq_thresholds = \
            <float32[:num_leq]>body.leq_thresholds_cbegin() if num_leq > 0 else None
        cdef uint32 num_gr = body.getNumGr()
        cdef const uint32[::1] gr_indices = <uint32[:num_gr]>body.gr_indices_cbegin() if num_gr > 0 else None
        cdef const float32[::1] gr_thresholds = <float32[:num_gr]>body.gr_thresholds_cbegin() if num_gr > 0 else None
        cdef uint32 num_eq = body.getNumEq()
        cdef const uint32[::1] eq_indices = <uint32[:num_eq]>body.eq_indices_cbegin() if num_eq > 0 else None
        cdef const float32[::1] eq_thresholds = <float32[:num_eq]>body.eq_thresholds_cbegin() if num_eq > 0 else None
        cdef uint32 num_neq = body.getNumNeq()
        cdef const uint32[::1] neq_indices = <uint32[:num_neq]>body.neq_indices_cbegin() if num_neq > 0 else None
        cdef const float32[::1] neq_thresholds = \
            <float32[:num_neq]>body.neq_thresholds_cbegin() if num_neq > 0 else None
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

    cdef __serialize_empty_body(self, const EmptyBodyImpl& body):
        cdef object body_state = None
        cdef object rule_state = [body_state, None]
        self.state.append(rule_state)

    cdef __serialize_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef uint32 num_leq = body.getNumLeq()
        cdef uint32 num_gr = body.getNumGr()
        cdef uint32 num_eq = body.getNumEq()
        cdef uint32 num_neq = body.getNumNeq()
        cdef object body_state = (np.asarray(<float32[:num_leq]>body.leq_thresholds_cbegin()) if num_leq > 0 else None,
                                  np.asarray(<uint32[:num_leq]>body.leq_indices_cbegin()) if num_leq > 0 else None,
                                  np.asarray(<float32[:num_gr]>body.gr_thresholds_cbegin()) if num_gr > 0 else None,
                                  np.asarray(<uint32[:num_gr]>body.gr_indices_cbegin()) if num_gr > 0 else None,
                                  np.asarray(<float32[:num_eq]>body.eq_thresholds_cbegin()) if num_eq > 0 else None,
                                  np.asarray(<uint32[:num_eq]>body.eq_indices_cbegin()) if num_eq > 0 else None,
                                  np.asarray(<float32[:num_neq]>body.neq_thresholds_cbegin()) if num_neq > 0 else None,
                                  np.asarray(<uint32[:num_neq]>body.neq_indices_cbegin()) if num_neq > 0 else None)
        cdef object rule_state = [body_state, None]
        self.state.append(rule_state)

    cdef __serialize_complete_head(self, const CompleteHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef object head_state = (np.asarray(<float64[:num_elements]>head.scores_cbegin()),)
        cdef object rule_state = self.state[len(self.state) - 1]
        rule_state[1] = head_state

    cdef __serialize_partial_head(self, const PartialHeadImpl& head):
        cdef uint32 num_elements = head.getNumElements()
        cdef object head_state = (np.asarray(<float64[:num_elements]>head.scores_cbegin()),
                                  np.asarray(<uint32[:num_elements]>head.indices_cbegin()))
        cdef object rule_state = self.state[len(self.state) - 1]
        rule_state[1] = head_state

    cdef unique_ptr[IBody] __deserialize_body(self, object body_state):
        if body_state is None:
            return unique_ptr[IBody]()
        else:
            return self.__deserialize_conjunctive_body(body_state)

    cdef unique_ptr[IBody] __deserialize_conjunctive_body(self, object body_state):
        cdef const float32[::1] leq_thresholds = body_state[0]
        cdef const uint32[::1] leq_indices = body_state[1]
        cdef const float32[::1] gr_thresholds = body_state[2]
        cdef const uint32[::1] gr_indices = body_state[3]
        cdef const float32[::1] eq_thresholds = body_state[4]
        cdef const uint32[::1] eq_indices = body_state[5]
        cdef const float32[::1] neq_thresholds = body_state[6]
        cdef const uint32[::1] neq_indices = body_state[7]
        cdef uint32 num_leq = leq_thresholds.shape[0] if leq_thresholds is not None else 0
        cdef uint32 num_gr = gr_thresholds.shape[0] if gr_thresholds is not None else 0
        cdef uint32 num_eq = eq_thresholds.shape[0] if eq_thresholds is not None else 0
        cdef uint32 num_neq = neq_thresholds.shape[0] if neq_thresholds is not None else 0
        cdef unique_ptr[ConjunctiveBodyImpl] body_ptr = make_unique[ConjunctiveBodyImpl](num_leq, num_gr, num_eq,
                                                                                         num_neq)
        cdef ConjunctiveBodyImpl.threshold_iterator threshold_iterator = body_ptr.get().leq_thresholds_begin()
        cdef ConjunctiveBodyImpl.index_iterator index_iterator = body_ptr.get().leq_indices_begin()
        cdef uint32 i

        for i in range(num_leq):
            threshold_iterator[i] = leq_thresholds[i]
            index_iterator[i] = leq_indices[i]

        threshold_iterator = body_ptr.get().gr_thresholds_begin()
        index_iterator = body_ptr.get().gr_indices_begin()

        for i in range(num_gr):
            threshold_iterator[i] = gr_thresholds[i]
            index_iterator[i] = gr_indices[i]

        threshold_iterator = body_ptr.get().eq_thresholds_begin()
        index_iterator = body_ptr.get().eq_indices_begin()

        for i in range(num_eq):
            threshold_iterator[i] = eq_thresholds[i]
            index_iterator[i] = eq_indices[i]

        threshold_iterator = body_ptr.get().neq_thresholds_begin()
        index_iterator = body_ptr.get().neq_indices_begin()

        for i in range(num_neq):
            threshold_iterator[i] = neq_thresholds[i]
            index_iterator[i] = neq_indices[i]

        return <unique_ptr[IBody]>move(body_ptr)

    cdef unique_ptr[IHead] __deserialize_head(self, object head_state):
        if len(head_state) > 1:
            return self.__deserialize_partial_head(head_state)
        else:
            return self.__deserialize_complete_head(head_state)

    cdef unique_ptr[IHead] __deserialize_complete_head(self, object head_state):
        cdef const float64[::1] scores = head_state[0]
        cdef uint32 num_elements = scores.shape[0]
        cdef unique_ptr[CompleteHeadImpl] head_ptr = make_unique[CompleteHeadImpl](num_elements)
        cdef CompleteHeadImpl.score_iterator score_iterator = head_ptr.get().scores_begin()
        cdef uint32 i

        for i in range(num_elements):
            score_iterator[i] = scores[i]

        return <unique_ptr[IHead]>move(head_ptr)

    cdef unique_ptr[IHead] __deserialize_partial_head(self, object head_state):
        cdef const float64[::1] scores = head_state[0]
        cdef const uint32[::1] indices = head_state[1]
        cdef uint32 num_elements = scores.shape[0]
        cdef unique_ptr[PartialHeadImpl] head_ptr = make_unique[PartialHeadImpl](num_elements)
        cdef PartialHeadImpl.score_iterator score_iterator = head_ptr.get().scores_begin()
        cdef PartialHeadImpl.index_iterator index_iterator = head_ptr.get().indices_begin()
        cdef uint32 i

        for i in range(num_elements):
            score_iterator[i] = scores[i]
            index_iterator[i] = indices[i]

        return <unique_ptr[IHead]>move(head_ptr)

    def contains_default_rule(self) -> bool:
        """
        Returns whether the model contains a default rule or not.

        :return: True, if the model contains a default rule, False otherwise
        """
        return self.rule_list_ptr.get().containsDefaultRule()

    def is_default_rule_taking_precedence(self) -> bool:
        """
        Returns whether the default rule takes precedence over the remaining rules or not.

        :return: True, if the default rule takes precedence over the remaining rules, False otherwise
        """
        return self.rule_list_ptr.get().isDefaultRuleTakingPrecedence()

    def visit(self, visitor: RuleModelVisitor):
        self.visitor = visitor
        self.rule_list_ptr.get().visit(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__visit_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))
        self.visitor = None

    def visit_used(self, visitor: RuleModelVisitor):
        self.visitor = visitor
        self.rule_list_ptr.get().visitUsed(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__visit_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))
        self.visitor = None

    def __reduce__(self):
        self.state = []
        self.rule_list_ptr.get().visit(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__serialize_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__serialize_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__serialize_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__serialize_partial_head))
        cdef bint default_rule_takes_precedence = self.rule_list_ptr.get().isDefaultRuleTakingPrecedence()
        cdef uint32 num_used_rules = self.rule_list_ptr.get().getNumUsedRules()
        cdef object state = (SERIALIZATION_VERSION, (self.state, default_rule_takes_precedence, num_used_rules))
        self.state = None
        return (RuleList, (), state)

    def __setstate__(self, state):
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError('Version of the serialized RuleModel is ' + str(version) + ', expected '
                                 + str(SERIALIZATION_VERSION))

        cdef object model_state = state[1]
        cdef list rule_list = model_state[0]
        cdef bint default_rule_takes_precedence = model_state[1]
        cdef uint32 num_rules = len(rule_list)
        cdef unique_ptr[IRuleList] rule_list_ptr = createRuleList(default_rule_takes_precedence)
        cdef object rule_state
        cdef unique_ptr[IBody] body_ptr
        cdef unique_ptr[IHead] head_ptr
        cdef uint32 i

        for i in range(num_rules):
            rule_state = rule_list[i]
            body_ptr = self.__deserialize_body(rule_state[0])
            head_ptr = self.__deserialize_head(rule_state[1])

            if body_ptr.get() == NULL:
                rule_list_ptr.get().addDefaultRule(move(head_ptr))
            else:
                rule_list_ptr.get().addRule(move(body_ptr), move(head_ptr))

        cdef uint32 num_used_rules = model_state[2]
        rule_list_ptr.get().setNumUsedRules(num_used_rules)
        self.rule_list_ptr = move(rule_list_ptr)
