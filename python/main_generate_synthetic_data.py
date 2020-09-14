#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Implements a method for generating synthetic multi-label datasets that allows to control the degree of marginal and
conditional dependence between the labels. The method originates from the paper "On label dependence and loss
minimization in multi-label classification", Dembczyński et al. 2012 (see
https://link.springer.com/article/10.1007/s10994-012-5285-8).

In the original paper five different types of synthetic datasets are used. To reproduce these datasets, the following
command line arguments must be specified:

(1) Marginal independence, conditional independence (Section 7.2):
    --marginal_independence=True
    --tau=0.0 (actually, the value of tau doesn't matter if marginal_independence=True)
    --p=0.1
    --dependent-error=False
    --one-error=False

(2) High marginal dependence, conditional independence (Section 7.3):
    --marginal_independence=False
    --tau=0.0
    --p=0.1
    --dependent-error=False
    --one-error=False

(3) Low marginal dependence, conditional independence (Section 7.3):
    --marginal_independence=False
    --tau=1.0
    --p=0.1
    --dependent-error=False
    --one-error=False

(4) Low marginal dependence, conditional dependence (Section 7.4):
    --marginal_independence=False
    --tau=1.0
    --p=0.1
    --dependent-error=True
    --one-error=False

(5) Low marginal dependency, extreme conditional dependence (Section 7.5):
    --marginal_independence=False
    --tau=1.0
    --p=0.0
    --dependent-error=False
    --one-error=True
"""
import logging as log

import numpy as np
from scipy.stats import bernoulli

from args import boolean_string, ArgumentParserBuilder
from boomer.data import save_data_set_and_meta_data
from runnables import Runnable


class GenerateSyntheticDataRunnable(Runnable):

    def _run(self, args):
        num_examples = args.num_examples
        num_labels = args.num_labels
        tau = args.tau
        p = args.p
        marginal_independence = args.marginal_independence
        dependent_error = args.dependent_error
        one_error = args.one_error
        features, labels = GenerateSyntheticDataRunnable.__generate_dataset(
            num_examples=num_examples, num_labels=num_labels, marginal_independence=marginal_independence, tau=tau, p=p,
            dependent_error=dependent_error, one_error=one_error, random_state=args.random_state)
        dataset_name = GenerateSyntheticDataRunnable.__get_dataset_name(
            num_examples=num_examples, num_labels=num_labels, tau=tau, p=p, dependent_error=dependent_error,
            one_error=one_error, marginal_independence=marginal_independence)
        meta_data = save_data_set_and_meta_data(args.output_dir, arff_file_name=dataset_name + '.arff',
                                                xml_file_name=dataset_name + '.xml', x=features, y=labels)
        log.info('The generated data set contains %s examples, %s attributes and %s labels', features.shape[0],
                 len(meta_data.attributes), len(meta_data.label_names))

    @staticmethod
    def __generate_dataset(num_examples: int, num_labels: int, tau: float, p: float, marginal_independence: bool,
                           dependent_error: bool, one_error: bool, random_state: int) -> (np.ndarray, np.ndarray):
        if num_examples < 1:
            raise ValueError('Invalid value given for parameter \'num-examples\': ' + str(num_examples))
        if num_labels < 1:
            raise ValueError('Invalid value given for parameter \'num_labels\': ' + str(num_labels))
        if tau < 0.0 or tau > 1.0:
            raise ValueError('Invalid value given for parameter \'tau\': ' + str(tau))
        if p < 0.0 or p > 1.0:
            raise ValueError('Invalid value given for parameter \'p\': ' + str(p))

        log.debug('Generating dataset...')

        # Initialize numpy's seed
        np.random.seed(random_state)

        if marginal_independence:
            # In this case, the features and labels are generated independently (cf. Dembczyński et al. 2012,
            # Section 7.2). This results in num_labels * 2 features.

            if tau > 0.0:
                log.warning('Parameter \'marginal-independence\' is True, parameter \'tau\' will be ignored...')
            if one_error:
                log.warning('Parameter \'marginal-independence\' is True, parameter \'one-error\' will be ignored...')
            if dependent_error:
                log.warning(
                    'Parameter \'marginal-independence\' is True, parameter \'dependent-error\' will be ignored...')

            x = np.empty((num_examples, num_labels * 2), dtype=float)
            y = np.empty((num_examples, num_labels), dtype=int)
            a = np.empty((num_labels, 2), dtype=float)

            for k in range(num_labels):
                current_x, current_y, current_a = GenerateSyntheticDataRunnable.__generate_features_and_labels(
                    num_examples=num_examples, num_labels=1, tau=1, p=p, dependent_error=False, one_error=False,
                    random_state=random_state)
                x[:, (k * 2, k * 2 + 1)] = current_x[:, :]
                y[:, k] = current_y[:, 0]
                a[k, :] = current_a
                random_state += 1
        else:
            x, y, _ = GenerateSyntheticDataRunnable.__generate_features_and_labels(
                num_examples=num_examples, num_labels=num_labels, tau=tau, p=p, dependent_error=dependent_error,
                one_error=one_error, random_state=random_state)

        log.info('Successfully generated dataset...')
        return x, y

    @staticmethod
    def __generate_features_and_labels(num_examples: int, num_labels: int, tau: float, p: float, dependent_error: bool,
                                       one_error: bool, random_state: int) -> (np.ndarray, np.ndarray, np.ndarray):
        # It is very important that this random number generator is used everywhere. If you want consistency, that
        # means, that a dataset with num_examples=n is a subset (actually the beginning) of a dataset with
        # num_examples > n, then you have to set the random states of the Bernoulli distribution to something different,
        # but never never to the same seed than here
        np.random.seed(random_state)

        # Randomly pick from the disk surrounding 0,0
        x = GenerateSyntheticDataRunnable.__disc_point_picking(num_examples)

        # Generate vector r, which is the variation from the base decision bound a = (1, 0), i.e., everything x = 0 is
        # +, everything x < 0 is -
        r = np.random.uniform(low=0, high=1, size=(num_labels, 2))

        # According to Dembczyński et al. 2012, Sec. 7.1
        a = tau * r
        a[:, 0] = 1 - a[:, 0]
        a = GenerateSyntheticDataRunnable.__normalize(a, axis=1)
        y = np.empty((num_examples, num_labels), dtype=int)

        if one_error:
            errors = np.zeros((num_examples, num_labels))

            if dependent_error:
                log.warning('Parameter \'one-error\' is True, parameter \'dependent-error\' will be ignored...')
            if p > 0.0:
                log.warning('Parameter \'one-error\' is True, parameter \'p\' will be ignored...')
        elif dependent_error:
            # Use explicitly None as random_state, which uses the singleton of np.random, i.e., it re-uses the number
            # generator which we initialized before with np.random.seed(random_state)
            errors = np.tile(bernoulli.rvs(p, size=num_examples, random_state=None).reshape(num_examples, 1),
                             num_labels)
        else:
            # Use explicitly None as random_state, which uses the singleton of np.random, i.e., it re-uses the number
            # generator which we initialized before with np.random.seed(random_state)
            errors = np.asarray(bernoulli.rvs(p, size=num_examples * num_labels, random_state=None)).reshape(
                (num_examples, num_labels))

        for r in range(num_examples):
            for c in range(num_labels):
                # Compute h_i(x) for each label and example
                label_set = a[c, 0] * x[r, 0] + a[c, 1] * x[r, 1] >= 0
                # Convert label into 1 / 0 notation
                label_value = 1 if label_set else 0
                # Error is either 0 or 1. If it is 0, nothing changes. If it is 1, then switch the label, which is done
                # by adding -1 if it was 1, or adding 1 if it was 0
                error = errors[r, c] * (-1 if label_set else 1)
                y[r, c] = label_value + error

        if one_error:
            # Flick one random label per example, according to Dembczyński et al. 2012, Sec. 7.5, this sets the Bayes
            # error for Hamming to 1 / m, m = number of labels (1 / m of label-example-combinations are wrong), and to
            # 1 - 1 / m for subset accuracy (because a label combination is changed in m different ways, so the bayes
            # classifier, even if learns the correct pattern on the instance space, it will make the correct prediction
            # for 1 / m examples with that label combinations, and hence (1 - 1 / m) the wrong prediction)
            flicked_label_indices = np.random.choice(np.arange(start=0, stop=num_labels, dtype=int), size=num_examples,
                                                     replace=True)

            for r in range(num_examples):
                c = flicked_label_indices[r]
                y[r, c] = 0 if y[r, c] > 0 else 1

        return x, y, a

    @staticmethod
    def __disc_point_picking(num_examples: int, radius: float = 1.0) -> np.ndarray:
        r = radius * np.sqrt(np.random.uniform(low=0, high=1, size=num_examples))
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num_examples)
        f1 = r * np.cos(theta)
        f2 = r * np.sin(theta)
        shape = (num_examples, 1)
        return np.hstack((f1.reshape(shape), f2.reshape(shape)))

    @staticmethod
    def __normalize(a: np.ndarray, order=2, axis=-1) -> np.ndarray:
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    @staticmethod
    def __get_dataset_name(num_examples: int, num_labels: int, tau: float, p: float, marginal_independence: bool,
                           dependent_error: bool, one_error: bool) -> str:
        name = 'synthetic_num-examples=' + str(num_examples) + '_num-labels=' + str(num_labels)

        if one_error:
            name += '_one-error'
        elif marginal_independence:
            name += '_marginal-independence_p=' + str(p)
        else:
            name += '_tau=' + str(tau) + '_p=' + str(p)

            if dependent_error:
                name += '_dependent-error'

        return name


if __name__ == '__main__':
    """
    Creates a synthetic data set with a specific number of examples and labels and saves it into a certain directory.
    The underlying model of such synthetic data set is defined as

    h_i(x) = 1, if a_i_1 * x_1 + a_i_2 + x_2 >= 0,
             0, otherwise

    where i corresponds to a certain label. The values of x_1 and x_2 are generated using unit disk point picking, i.e., 
    they are uniformly drawn from a circle with radius 1. The values a_i_1 and a_i_2 are drawn randomly in order to 
    model different degrees of similarity between the labels of an instance. They are controlled by a value tau 
    (in [0, 1]) in the following way:

    a_i_1 = 1 - tau * r_1
    a_i_2 = tau * r_2

    where r_1 and r_2 (in [0, 1]) are drawn randomly from the uniform distribution. The parameters a_i are normalized to 
    satisfy ||a_i||_2 = 1.

    Additionally, each label can be disturbed by adding a random error term that follows a Bernoulli distribution:

    e_i(x) = -Ber(p), if a_i_1 * x_1 + a_i_2 * x_2 >= 0,
              Ber(p), otherwise

    When setting the parameter \'marginal-independence\' to True, the produce is performed for each label independently,
    resulting in a data set with 2 * num_labels attributes where the labels are marginally independent from each other.
    The parameters \'tau\', \'dependent-error\' and \'one-error\' will be ignored in such case.

    When setting the parameter \'one-error\' to True, the data set will be first generated without any random errors and
    afterwards one random label of each example is flicked to the opposite. This results in an extreme form of 
    conditional dependence between the labels. The parameters \'p\' and \'dependent-error\' will be ignored in such 
    case.

    When setting the parameter \'dependent-error\' to True, the random Bernoulli variables that are used for computing 
    the error terms to disturb the labels of a certain example are chosen to be the same. This results in a data set 
    where the labels are conditionally dependent on each other.
    """
    parser = ArgumentParserBuilder(description='Allows to generate synthetic multi-label data sets') \
        .add_random_state_argument() \
        .build()
    parser.add_argument('--output-dir', type=str,
                        help='The path of the directory into which generated dataset should be written')
    parser.add_argument('--num-examples', type=int, default=1000, help='The number of examples to be generated')
    parser.add_argument('--num-labels', type=int, default=10, help='The number of labels in the dataset')
    parser.add_argument('--marginal-independence', type=boolean_string, default=False,
                        help='True, if the labels should be marginally independent, False otherwise')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='The value of the parameter \'tau\' that controls the degree of marginal dependency')
    parser.add_argument('--p', type=float, default=0.1,
                        help='The value of the Bernoulli parameter \'p\' that controls the Bayes error rate')
    parser.add_argument('--dependent-error', type=boolean_string, default=False,
                        help='True, if the error for the labels of an example should be the same, False otherwise')
    parser.add_argument('--one-error', type=boolean_string, default=False,
                        help='True, if exactly one error should be made per example, False otherwise')
    runnable = GenerateSyntheticDataRunnable()
    runnable.run(parser)
