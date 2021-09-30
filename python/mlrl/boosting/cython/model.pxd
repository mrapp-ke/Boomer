from mlrl.common.cython.model cimport IModelBuilder, ModelBuilder


cdef extern from "boosting/model/rule_list.hpp" namespace "boosting" nogil:

    cdef cppclass RuleListBuilderImpl"boosting::RuleListBuilder"(IModelBuilder):
        pass


cdef class RuleListBuilder(ModelBuilder):
    pass
