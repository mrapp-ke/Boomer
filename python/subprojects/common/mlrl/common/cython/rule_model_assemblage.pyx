"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.input cimport NominalFeatureMask, FeatureMatrix, LabelMatrix
from mlrl.common.cython.model cimport ModelBuilder, RuleModel

from cython.operator cimport dereference

from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class RuleModelAssemblage:
    """
    A wrapper for the pure virtual C++ class `IRuleModelAssemblage`.
    """

    def induce_rules(self, NominalFeatureMask nominal_feature_mask not None, FeatureMatrix feature_matrix not None,
                     LabelMatrix label_matrix not None, int random_state,
                     ModelBuilder model_builder not None) -> RuleModel:
        cdef unique_ptr[RuleModelImpl] rule_model_ptr = self.rule_model_assemblage_ptr.get().induceRules(
            dereference(nominal_feature_mask.nominal_feature_mask_ptr), dereference(feature_matrix.feature_matrix_ptr),
            dereference(label_matrix.label_matrix_ptr), random_state,
            dereference(model_builder.model_builder_ptr))
        cdef RuleModel model = RuleModel.__new__(RuleModel)
        model.model_ptr = move(rule_model_ptr)
        return model


cdef class SequentialRuleModelAssemblageFactory(RuleModelAssemblageFactory):
    """
    A wrapper for the C++ class `SequentialRuleModelAssemblageFactory`.
    """

    def __cinit__(self):
        self.rule_model_assemblage_factory_ptr = <unique_ptr[IRuleModelAssemblageFactory]>make_unique[SequentialRuleModelAssemblageFactoryImpl]()
