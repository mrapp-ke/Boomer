"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.algorithm cimport copy
from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from _io import StringIO

import numpy as np

SERIALIZATION_VERSION = 1


cdef class RuleModel:
    """
    A wrapper for the C++ class `RuleModel`.
    """

    def get_num_rules(self) -> int:
        return self.model_ptr.get().getNumRules()

    def get_num_used_rules(self) -> int:
        return self.model_ptr.get().getNumUsedRules()

    def set_num_used_rules(self, int num_used_rules):
        self.model_ptr.get().setNumUsedRules(num_used_rules)

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


cdef uint32 __format_conditions(uint32 num_processed_conditions, uint32 num_conditions,
                                ConjunctiveBodyImpl.index_const_iterator index_iterator,
                                ConjunctiveBodyImpl.threshold_const_iterator threshold_iterator, object attributes,
                                bint print_feature_names, bint print_nominal_values, object text, object comparator):
    cdef uint32 result = num_processed_conditions
    cdef uint32 feature_index, i
    cdef float32 threshold
    cdef object attribute

    for i in range(num_conditions):
        if result > 0:
            text.write(' & ')

        feature_index = index_iterator[i]
        threshold = threshold_iterator[i]
        attribute = attributes[feature_index] if len(attributes) > feature_index else None

        if print_feature_names and attribute is not None:
            text.write(attribute.attribute_name)
        else:
            text.write(str(feature_index))

        text.write(' ')
        text.write(comparator)
        text.write(' ')

        if attribute is not None and attribute.nominal_values is not None:
            if print_nominal_values and len(attribute.nominal_values) > threshold:
                text.write('"' + attribute.nominal_values[<uint32>threshold] + '"')
            else:
                text.write(str(<uint32>threshold))
        else:
            text.write(str(threshold))

        result += 1

    return result


cdef class RuleModelFormatter:
    """
    Allows to create textual representations of the rules that are contained by a `RuleModel`.
    """

    def __cinit__(self, list attributes not None, list labels not None, bint print_feature_names,
                  bint print_label_names, bint print_nominal_values):
        """
        :param attributes:              A list that contains the attributes
        :param labels:                  A list that contains the labels
        :param print_feature_names:     True, if the names of features should be printed, False otherwise
        :param print_label_names:       True, if the names of labels should be printed, False otherwise
        :param print_nominal_values:    True, if the values of nominal values should be printed, False otherwise
        """
        self.print_feature_names = print_feature_names
        self.print_label_names = print_label_names
        self.print_nominal_values = print_nominal_values
        self.attributes = attributes
        self.labels = labels
        self.text = StringIO()

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        self.text.write('{}')

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef object text = self.text
        cdef bint print_feature_names = self.print_feature_names
        cdef bint print_nominal_values = self.print_nominal_values
        cdef list attributes = self.attributes
        cdef uint32 num_processed_conditions = 0

        text.write('{')

        cdef ConjunctiveBodyImpl.threshold_const_iterator threshold_iterator = body.leq_thresholds_cbegin()
        cdef ConjunctiveBodyImpl.index_const_iterator index_iterator = body.leq_indices_cbegin()
        cdef uint32 num_conditions = body.getNumLeq()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '<=')

        threshold_iterator = body.gr_thresholds_cbegin()
        index_iterator = body.gr_indices_cbegin()
        num_conditions = body.getNumGr()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '>')

        threshold_iterator = body.eq_thresholds_cbegin()
        index_iterator = body.eq_indices_cbegin()
        num_conditions = body.getNumEq()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '==')

        threshold_iterator = body.neq_thresholds_cbegin()
        index_iterator = body.neq_indices_cbegin()
        num_conditions = body.getNumNeq()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '!=')

        text.write('}')

    cdef __visit_complete_head(self, const CompleteHeadImpl& head):
        cdef object text = self.text
        cdef bint print_label_names = self.print_label_names
        cdef list labels = self.labels
        cdef CompleteHeadImpl.score_const_iterator score_iterator = head.scores_cbegin()
        cdef uint32 num_elements = head.getNumElements()
        cdef uint32 i

        text.write(' => (')

        for i in range(num_elements):
            if i > 0:
                text.write(', ')

            if print_label_names and len(labels) > i:
                text.write(labels[i].attribute_name)
            else:
                text.write(str(i))

            text.write(' = ')
            text.write('{0:.2f}'.format(score_iterator[i]))

        text.write(')\n')

    cdef __visit_partial_head(self, const PartialHeadImpl& head):
        cdef object text = self.text
        cdef bint print_label_names = self.print_label_names
        cdef list labels = self.labels
        cdef PartialHeadImpl.score_const_iterator score_iterator = head.scores_cbegin()
        cdef PartialHeadImpl.index_const_iterator index_iterator = head.indices_cbegin()
        cdef uint32 num_elements = head.getNumElements()
        cdef uint32 label_index, i

        text.write(' => (')

        for i in range(num_elements):
            if i > 0:
                text.write(', ')

            label_index = index_iterator[i]

            if print_label_names and len(labels) > label_index:
                text.write(labels[label_index].attribute_name)
            else:
                text.write(str(label_index))

            text.write(' = ')
            text.write('{0:.2f}'.format(score_iterator[i]))

        text.write(')\n')

    def format(self, RuleModel model):
        """
        Creates a textual representation of a specific model.

        :param model: The `RuleModel` to be formatted
        """
        model.model_ptr.get().visitUsed(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapCompleteHeadVisitor(<void*>self, <CompleteHeadCythonVisitor>self.__visit_complete_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))

    def get_text(self) -> object:
        """
        Returns the textual representation that has been created via the `format` method.

        :return: The textual representation
        """
        return self.text.getvalue()
