.. _rules:

Understanding the Rules Language
--------------------------------

The top of each rule file has the "default rule", a special rule without any conditions on features and therefore applies to any given example. For each label, it provides a "default prediction" or "starting value". The prediction for a particular label is positive, if most examples are associated with the respective label, otherwise it is negative. The ratio between the number of examples that are associated with a label and those that are not affects the absolute size of the default prediction. Values that are farther away from zero indicate that the data is very unbalanced.

For example default rule:

    ``{} => (label1 = -1.45, label2 = -1.45, label3 = -1.89, label3 = -1.94)``

The rest of the rules in the file. unlike the default rule, do only apply to examples that satisfy the rules' conditions. If a rule applies to an example, the example is said to be "covered" by the rule. In this case, the rule assigns a positive or negative value to one or several labels. The number of labels to be considered by a rule is controllable via the parameter `head_type` (see :ref:`parameters<Parameters>` below). A positive value expresses a preference towards predicting the corresponding label as relevant. A negative value contributes towards predicting the label as irrelevant. The absolute size of the value corresponds to the weight of the rule's prediction. The larger the value, the stronger the impact of the respective rule, compared to other ones.
 
A more formal description of the rules is given in Section 2.2. "Multi-label Classification Rules" of :ref:`our ECML paper<firstpartyreferences>`.

Example rule:

    ``{feature1 <= 1.53 & feature2 <= 7.935 & feature3 <= 6.604 & feature4 <= 74.81 & feature5 <= 12.395 & feature6 <= 0.305 & feature7 <= 0.415 & feature8 <= 6.83} => (label1= -0.31)``

    (when this rule applies, it has a negative contribution towards `label1`.)

The default prediction strategy thus use the rule's prediction value which can be either positive, zero or negative and the prediction is `1` for all values > 0 and `0` otherwise. This strategy aims to optimize the Hamming loss measure. There is also another strategy for optimizing the Subset 0/1 loss. You can specify which strategy should be used via the :ref:`parameter<Parameters>` `predictor`. A description of both prediction strategies can be found in Section 4.3 "Prediction" of :ref:`our ECML paper<firstpartyreferences>`.
 
There is also a probabilistic interpretation of the aggregated value that is obtained for an individual label and example: By default, the (label-wise) logistic loss function is used for training (see :ref:`parameter<Parameters>` `loss` below). In this case, the aggregated value can be viewed as log odds. Passing it through the `logistic sigmoid function<https://en.wikipedia.org/wiki/Sigmoid_function>` results in a value between 0 and 1 that corresponds to the probability that the respective label is relevant to the example at hand.
