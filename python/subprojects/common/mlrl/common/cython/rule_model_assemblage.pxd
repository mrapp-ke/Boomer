from libcpp cimport bool


cdef extern from "common/rule_induction/rule_model_assemblage_sequential.hpp" nogil:

    cdef cppclass ISequentialRuleModelAssemblageConfig:

        # Functions:

        bool getUseDefaultRule() const

        ISequentialRuleModelAssemblageConfig& setUseDefaultRule(bool useDefaultRule) except +


cdef class SequentialRuleModelAssemblageConfig:

    # Attributes:

    cdef ISequentialRuleModelAssemblageConfig* config_ptr
