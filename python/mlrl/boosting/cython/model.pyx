"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class RuleListBuilder(ModelBuilder):
    """
    A wrapper for the C++ class `RuleListBuilder`.
    """

    def __cinit__(self):
        self.model_builder_ptr = <shared_ptr[IModelBuilder]>make_shared[RuleListBuilderImpl]()
