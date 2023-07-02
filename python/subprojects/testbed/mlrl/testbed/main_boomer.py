"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.boosting_learners import Boomer
from mlrl.boosting.config import BOOSTING_RULE_LEARNER_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerConfig

from mlrl.testbed.runnables import RuleLearnerRunnable


def main():
    RuleLearnerRunnable(description='Allows to run experiments using the BOOMER algorithm',
                        learner_name='boomer',
                        learner_type=Boomer,
                        config_type=BoomerConfig,
                        parameters=BOOSTING_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
