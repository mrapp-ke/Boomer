from mlrl.common.cython._types cimport uint32, float32, float64

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.list cimport list as double_linked_list


cdef extern from "common/model/body.hpp" nogil:

    cdef cppclass IBody:
        pass


cdef extern from "common/model/body_empty.hpp" nogil:

    cdef cppclass EmptyBodyImpl"EmptyBody"(IBody):
        pass


cdef extern from "common/model/body_conjunctive.hpp" nogil:

    cdef cppclass ConjunctiveBodyImpl"ConjunctiveBody"(IBody):

        ConjunctiveBodyImpl(uint32 numLeq, uint32 numGr, uint32 numEq, uint32 numNeq) except +

        ctypedef float32* threshold_iterator

        ctypedef const float32* threshold_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        uint32 getNumLeq()

        threshold_iterator leq_thresholds_begin()

        threshold_iterator leq_thresholds_end()

        threshold_const_iterator leq_thresholds_cbegin()

        threshold_const_iterator leq_thresholds_cend()

        index_iterator leq_indices_begin()

        index_iterator leq_indices_end()

        index_const_iterator leq_indices_cbegin()

        index_const_iterator leq_indices_cend()

        uint32 getNumGr()

        threshold_iterator gr_thresholds_begin()

        threshold_iterator gr_thresholds_end()

        threshold_const_iterator gr_thresholds_cbegin()

        threshold_const_iterator gr_thresholds_cend()

        index_iterator gr_indices_begin()

        index_iterator gr_indices_end()

        index_const_iterator gr_indices_cbegin()

        index_const_iterator gr_indices_cend()

        uint32 getNumEq()

        threshold_iterator eq_thresholds_begin()

        threshold_iterator eq_thresholds_end()

        threshold_const_iterator eq_thresholds_cbegin()

        threshold_const_iterator eq_thresholds_cend()

        index_iterator eq_indices_begin()

        index_iterator eq_indices_end()

        index_const_iterator eq_indices_cbegin()

        index_const_iterator eq_indices_cend()

        uint32 getNumNeq()

        threshold_iterator neq_thresholds_begin()

        threshold_iterator neq_thresholds_end()

        threshold_const_iterator neq_thresholds_cbegin()

        threshold_const_iterator neq_thresholds_cend()

        index_iterator neq_indices_begin()

        index_iterator neq_indices_end()

        index_const_iterator neq_indices_cbegin()

        index_const_iterator neq_indices_cend()


ctypedef void (*EmptyBodyVisitor)(const EmptyBodyImpl&)

ctypedef void (*ConjunctiveBodyVisitor)(const ConjunctiveBodyImpl&)


cdef extern from "common/model/head.hpp" nogil:

    cdef cppclass IHead:
        pass


cdef extern from "common/model/head_full.hpp" nogil:

    cdef cppclass FullHeadImpl"FullHead"(IHead):

        FullHeadImpl(uint32 numElements) except +

        ctypedef float64* score_iterator

        ctypedef const float64* score_const_iterator

        uint32 getNumElements()

        score_iterator scores_begin()

        score_iterator scores_end()

        score_const_iterator scores_cbegin()

        score_const_iterator scores_cend()


cdef extern from "common/model/head_partial.hpp" nogil:

    cdef cppclass PartialHeadImpl"PartialHead"(IHead):

        PartialHeadImpl(uint32 numElements) except +

        ctypedef float64* score_iterator

        ctypedef const float64* score_const_iterator

        ctypedef uint32* index_iterator

        ctypedef const uint32* index_const_iterator

        uint32 getNumElements()

        score_iterator scores_begin()

        score_iterator scores_end()

        score_const_iterator scores_cbegin()

        score_const_iterator scores_cend()

        index_iterator indices_begin()

        index_iterator indices_end()

        index_const_iterator indices_cbegin()

        index_const_iterator indices_cend()


ctypedef void (*FullHeadVisitor)(const FullHeadImpl&)

ctypedef void (*PartialHeadVisitor)(const PartialHeadImpl&)


cdef extern from "common/model/rule.hpp" nogil:

    cdef cppclass RuleImpl"Rule":

        const IBody& getBody()

        const IHead& getHead()


cdef extern from "common/model/rule_model.hpp" nogil:

    cdef cppclass UsedIterator"RuleModel::UsedIterator":

        const RuleImpl& operator*()

        UsedIterator& operator++()

        UsedIterator& operator++(int n)

        bool operator!=(const UsedIterator& rhs)


    cdef cppclass RuleModelImpl"RuleModel":

        ctypedef double_linked_list[RuleImpl].const_iterator const_iterator

        ctypedef UsedIterator used_const_iterator

        const_iterator cbegin()

        const_iterator cend()

        used_const_iterator used_cbegin()

        used_const_iterator used_cend()

        uint32 getNumRules()

        uint32 getNumUsedRules()

        void setNumUsedRules(uint32 numUsedRules)

        void addRule(unique_ptr[IBody] bodyPtr, unique_ptr[IHead] headPtr)

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor)

        void visitUsed(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                       FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor)


cdef extern from "common/model/model_builder.hpp" nogil:

    cdef cppclass IModelBuilder:
        pass


cdef extern from *:
    """
    #include "common/model/body.hpp"
    #include "common/model/head.hpp"


    typedef void (*EmptyBodyCythonVisitor)(void*, const EmptyBody&);

    typedef void (*ConjunctiveBodyCythonVisitor)(void*, const ConjunctiveBody&);

    typedef void (*FullHeadCythonVisitor)(void*, const FullHead&);

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

    static inline IHead::FullHeadVisitor wrapFullHeadVisitor(void* self, FullHeadCythonVisitor visitor) {
        return [=](const FullHead& head) {
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

    ctypedef void (*FullHeadCythonVisitor)(void*, const FullHeadImpl&)

    ctypedef void (*PartialHeadCythonVisitor)(void*, const PartialHeadImpl&)

    EmptyBodyVisitor wrapEmptyBodyVisitor(void* self, EmptyBodyCythonVisitor visitor)

    ConjunctiveBodyVisitor wrapConjunctiveBodyVisitor(void* self, ConjunctiveBodyCythonVisitor visitor)

    FullHeadVisitor wrapFullHeadVisitor(void* self, FullHeadCythonVisitor visitor)

    PartialHeadVisitor wrapPartialHeadVisitor(void* self, PartialHeadCythonVisitor visitor)


cdef class RuleModel:

    # Attributes:

    cdef unique_ptr[RuleModelImpl] model_ptr

    # Functions:

    cpdef int get_num_rules(self)

    cpdef int get_num_used_rules(self)

    cpdef object set_num_used_rules(self, uint32 num_used_rules)


cdef class ModelBuilder:

    # Attributes:

    cdef shared_ptr[IModelBuilder] model_builder_ptr


cdef class RuleModelSerializer:

    # Attributes:

    cdef list state

    # Functions:

    cdef __visit_empty_body(self, const EmptyBodyImpl& body)

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body)

    cdef __visit_full_head(self, const FullHeadImpl& head)

    cdef __visit_partial_head(self, const PartialHeadImpl& head)

    cpdef object serialize(self, RuleModel model)

    cpdef deserialize(self, RuleModel model, object state)


cdef class RuleModelFormatter:

    # Attributes:

    cdef list attributes

    cdef list labels

    cdef bint print_feature_names

    cdef bint print_label_names

    cdef bint print_nominal_values

    cdef object text

    # Functions:

    cdef __visit_empty_body(self, const EmptyBodyImpl& body)

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body)

    cdef __visit_full_head(self, const FullHeadImpl& head)

    cdef __visit_partial_head(self, const PartialHeadImpl& head)

    cpdef void format(self, RuleModel model)

    cpdef object get_text(self)
