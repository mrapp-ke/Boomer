"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.seco.cython.heuristics cimport Heuristic

from libcpp.memory cimport make_shared


cdef class LabelWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseRuleEvaluationFactory`.
    """
    pass


cdef class HeuristicLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `HeuristicLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, Heuristic heuristic, bint predictMajority = False):
        """
        :param heuristic:       The heuristic that should be used
        :param predictMajority: True, if for each label the majority label should be predicted, False, if the minority
                                label should be predicted
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[HeuristicLabelWiseRuleEvaluationFactoryImpl](
            heuristic.heuristic_ptr, predictMajority)
