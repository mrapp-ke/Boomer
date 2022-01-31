"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class SequentialRuleModelAssemblageConfig:
    """
    Allows to configure an algorithm that sequentially induces several rules, optionally starting with a default rule,
    that are added to a rule-based model.
    """

    def get_use_default_rule(self) -> bool:
        """
        Returns whether a default rule should be used or not.

        :return: True, if a default rule should be used, False otherwise
        """
        return self.config_ptr.getUseDefaultRule()

    def set_use_default_rule(self, use_default_rule: bool) -> SequentialRuleModelAssemblageConfig:
        """
        Sets whether a default rule should be used or not.

        :param use_default_rule:    True, if a default rule should be used, False otherwise
        :return:                    A `SequentialRuleModelAssemblageConfig` that allows further configuration of the
                                    algorithm for the induction of several rules that are added to a rule-based model
        """
        self.config_ptr.setUseDefaultRule(use_default_rule)
        return self
