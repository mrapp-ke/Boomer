#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides learners based on XGBoost.
"""
from math import log2, floor

from xgboost import XGBClassifier

from boomer.learners import Learner


class XGBoost(Learner):
    """
    A configuration of XGBoost using decision trees as ensemble members.
    """

    def __init__(self, learning_rate: float = 1.0, reg_lambda: float = 0.0, objective: str = 'binary:logistic'):
        """
        :param learning_rate:   The learning rate
        :param reg_lambda:      The L2 regularization weight
        :param objective:       The objective function to be minimized
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.xgboost = XGBClassifier(booster='gbtree', tree_method='exact', subsample=0.66, colsample_bytree=1.0,
                                     colsample_bylevel=1.0, reg_alpha=0.0, n_jobs=1)

    def fit(self, x, y):
        num_features = x.shape[1]
        colsample_bynode = float(floor(log2(num_features - 1) + 1)) / float(num_features)

        xgboost = self.xgboost
        xgboost.random_state = self.random_state
        xgboost.learning_rate = float(self.learning_rate)
        xgboost.objective = str(self.objective)
        xgboost.reg_lambda = float(self.reg_lambda)
        xgboost.colsample_bynode = colsample_bynode
        xgboost.fit(x, y)
        self.classes_ = xgboost.classes_
        return self

    def predict(self, x):
        return self.xgboost.predict(x)

    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'reg_lambda': self.reg_lambda,
            'objective': self.objective
        }

    def get_name(self) -> str:
        return 'xgboost_objective=' + str(self.objective) + '_learning-rate=' + str(
            self.learning_rate) + '_reg-lambda=' + str(self.reg_lambda)
