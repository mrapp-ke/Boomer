cimport numpy as npc

from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, float64, uint32


cdef extern from "common/model/body.hpp" nogil:

    cdef cppclass IBody:
        pass


cdef extern from "common/model/body_empty.hpp" nogil:

    cdef cppclass EmptyBodyImpl"EmptyBody"(IBody):
        pass


cdef extern from "common/model/body_conjunctive.hpp" nogil:

    cdef cppclass ConjunctiveBodyImpl"ConjunctiveBody"(IBody):

        ctypedef float32* threshold_iterator

        ctypedef const float32* threshold_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        # Constructors:

        ConjunctiveBodyImpl(uint32 numLeq, uint32 numGr, uint32 numEq, uint32 numNeq)

        # Functions:

        uint32 getNumLeq() const

        threshold_iterator leq_thresholds_begin()

        threshold_const_iterator leq_thresholds_cbegin() const

        index_iterator leq_indices_begin()

        index_const_iterator leq_indices_cbegin() const

        uint32 getNumGr() const

        threshold_iterator gr_thresholds_begin()

        threshold_const_iterator gr_thresholds_cbegin() const

        index_iterator gr_indices_begin()

        index_const_iterator gr_indices_cbegin() const

        uint32 getNumEq() const

        threshold_iterator eq_thresholds_begin()

        threshold_const_iterator eq_thresholds_cbegin() const

        index_iterator eq_indices_begin()

        index_const_iterator eq_indices_cbegin() const

        uint32 getNumNeq() const

        threshold_iterator neq_thresholds_begin()

        threshold_const_iterator neq_thresholds_cbegin() const

        index_iterator neq_indices_begin()

        index_const_iterator neq_indices_cbegin() const


cdef extern from "common/model/head.hpp" nogil:

    cdef cppclass IHead:
        pass


cdef extern from "common/model/head_complete.hpp" nogil:

    cdef cppclass CompleteHeadImpl"CompleteHead"(IHead):

        ctypedef float64* score_iterator

        ctypedef const float64* score_const_iterator

        # Functions:

        uint32 getNumElements() const

        score_iterator scores_begin()

        score_const_iterator scores_cbegin() const


cdef extern from "common/model/head_partial.hpp" nogil:

    cdef cppclass PartialHeadImpl"PartialHead"(IHead):

        ctypedef float64* score_iterator

        ctypedef const float64* score_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        # Functions:

        uint32 getNumElements() const

        score_iterator scores_begin()

        score_const_iterator scores_cbegin() const

        index_iterator indices_begin()

        index_const_iterator indices_cbegin() const


ctypedef void (*EmptyBodyVisitor)(const EmptyBodyImpl&)

ctypedef void (*ConjunctiveBodyVisitor)(const ConjunctiveBodyImpl&)

ctypedef void (*CompleteHeadVisitor)(const CompleteHeadImpl&)

ctypedef void (*PartialHeadVisitor)(const PartialHeadImpl&)


cdef extern from "common/model/rule_model.hpp" nogil:

    cdef cppclass IRuleModel:

        # Functions:

        uint32 getNumRules() const

        uint32 getNumUsedRules() const

        void setNumUsedRules(uint32 numUsedRules)


cdef extern from "common/model/rule_list.hpp" nogil:

    cdef cppclass IRuleList(IRuleModel):

        # Functions:

        void addDefaultRule(unique_ptr[IHead] headPtr)

        void addRule(unique_ptr[IBody] bodyPtr, unique_ptr[IHead] headPtr)

        bool containsDefaultRule() const

        bool isDefaultRuleTakingPrecedence() const

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const

        void visitUsed(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                       CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const


    unique_ptr[IRuleList] createRuleList(bool defaultRuleTakesPrecedence)


ctypedef IRuleList* RuleListPtr


cdef extern from *:
    """
    #include "common/model/body.hpp"
    #include "common/model/head.hpp"


    typedef void (*EmptyBodyCythonVisitor)(void*, const EmptyBody&);

    typedef void (*ConjunctiveBodyCythonVisitor)(void*, const ConjunctiveBody&);

    typedef void (*CompleteHeadCythonVisitor)(void*, const CompleteHead&);

    typedef void (*PartialHeadCythonVisitor)(void*, const PartialHead&);

    static inline IBody::EmptyBodyVisitor wrapEmptyBodyVisitor(void* self, EmptyBodyCythonVisitor visitor) {
        return [=](const EmptyBody& body) {
            visitor(self, body);
        };
    }

    static inline IBody::ConjunctiveBodyVisitor wrapConjunctiveBodyVisitor(void* self,
                                                                           ConjunctiveBodyCythonVisitor visitor) {
        return [=](const ConjunctiveBody& body) {
            visitor(self, body);
        };
    }

    static inline IHead::CompleteHeadVisitor wrapCompleteHeadVisitor(void* self, CompleteHeadCythonVisitor visitor) {
        return [=](const CompleteHead& head) {
            visitor(self, head);
        };
    }

    static inline IHead::PartialHeadVisitor wrapPartialHeadVisitor(void* self, PartialHeadCythonVisitor visitor) {
        return [=](const PartialHead& head) {
            visitor(self, head);
        };
    }
    """

    ctypedef void (*EmptyBodyCythonVisitor)(void*, const EmptyBodyImpl&)

    ctypedef void (*ConjunctiveBodyCythonVisitor)(void*, const ConjunctiveBodyImpl&)

    ctypedef void (*CompleteHeadCythonVisitor)(void*, const CompleteHeadImpl&)

    ctypedef void (*PartialHeadCythonVisitor)(void*, const PartialHeadImpl&)

    EmptyBodyVisitor wrapEmptyBodyVisitor(void* self, EmptyBodyCythonVisitor visitor)

    ConjunctiveBodyVisitor wrapConjunctiveBodyVisitor(void* self, ConjunctiveBodyCythonVisitor visitor)

    CompleteHeadVisitor wrapCompleteHeadVisitor(void* self, CompleteHeadCythonVisitor visitor)

    PartialHeadVisitor wrapPartialHeadVisitor(void* self, PartialHeadCythonVisitor visitor)


cdef class EmptyBody:
    pass


cdef class ConjunctiveBody:

    # Attributes:

    cdef readonly npc.ndarray leq_indices

    cdef readonly npc.ndarray leq_thresholds

    cdef readonly npc.ndarray gr_indices

    cdef readonly npc.ndarray gr_thresholds

    cdef readonly npc.ndarray eq_indices

    cdef readonly npc.ndarray eq_thresholds

    cdef readonly npc.ndarray neq_indices

    cdef readonly npc.ndarray neq_thresholds


cdef class CompleteHead:

    # Attributes:

    cdef readonly npc.ndarray scores


cdef class PartialHead:

    # Attributes:

    cdef readonly npc.ndarray indices

    cdef readonly npc.ndarray scores


cdef class RuleModel:

    # Functions:

    cdef IRuleModel* get_rule_model_ptr(self)


cdef class RuleList(RuleModel):

    # Attributes:

    cdef unique_ptr[IRuleList] rule_list_ptr

    cdef object visitor

    cdef object state

    # Functions:

    cdef __visit_empty_body(self, const EmptyBodyImpl& body)

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body)

    cdef __visit_complete_head(self, const CompleteHeadImpl& head)

    cdef __visit_partial_head(self, const PartialHeadImpl& head)

    cdef __serialize_empty_body(self, const EmptyBodyImpl& body)

    cdef __serialize_conjunctive_body(self, const ConjunctiveBodyImpl& body)

    cdef __serialize_complete_head(self, const CompleteHeadImpl& head)

    cdef __serialize_partial_head(self, const PartialHeadImpl& head)

    cdef unique_ptr[IBody] __deserialize_body(self, object body_state)

    cdef unique_ptr[IBody] __deserialize_conjunctive_body(self, object body_state)

    cdef unique_ptr[IHead] __deserialize_head(self, object head_state)

    cdef unique_ptr[IHead] __deserialize_complete_head(self, object head_state)

    cdef unique_ptr[IHead] __deserialize_partial_head(self, object head_state)


cdef inline RuleModel create_rule_model(unique_ptr[IRuleModel] rule_model_ptr):
    cdef IRuleModel* ptr = rule_model_ptr.release()
    cdef IRuleList* rule_list_ptr = dynamic_cast[RuleListPtr](ptr)
    cdef RuleList rule_list

    if rule_list_ptr != NULL:
        rule_list = RuleList.__new__(RuleList)
        rule_list.rule_list_ptr = unique_ptr[IRuleList](rule_list_ptr)
        return rule_list
    else:
        del ptr
        raise RuntimeError('Encountered unsupported IRuleModel object')
