"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class RuleListBuilder(ModelBuilder):
    """
    A wrapper for the C++ class `RuleListBuilder`.
    """

    def __cinit__(self):
        self.model_builder_ptr = <unique_ptr[IModelBuilder]>make_unique[RuleListBuilderImpl]()
