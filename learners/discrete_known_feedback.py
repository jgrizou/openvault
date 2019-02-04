import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import random
import operator
import itertools

import scipy.stats
import numpy as np


class DiscreteKnownFeedbackLearner(object):

    def __init__(self, n_hypothesis):
        self.n_hypothesis = n_hypothesis

        self.hypothesis_scores = np.ones(self.n_hypothesis, dtype=np.int)

        self.feedback_history = []

        self.all_equal_flash_patterns = self.get_all_equal_flash_patterns()

    def get_all_equal_flash_patterns(self):
        all_equal_flash_patterns = []
        for comb in itertools.combinations(range(self.n_hypothesis), self.n_hypothesis//2):
            flash_pattern = np.zeros(self.n_hypothesis,dtype=np.int)
            flash_pattern[np.array(comb)] = 1
            all_equal_flash_patterns.append(flash_pattern)
        return all_equal_flash_patterns

    def get_next_flash_pattern(self, method='equal_uncertainty'):
        if method == 'random':
            return np.random.randint(0, 2, self.n_hypothesis)
        elif method == 'equal_random':
            return random.choice(self.all_equal_flash_patterns)
        elif method == 'equal_uncertainty':
            uncertain_equal_flash_patterns = self.get_uncertain_equal_flash_patterns()
            return random.choice(uncertain_equal_flash_patterns)
        else:
            raise Exception('{} not defined'.format(method))

    def get_uncertain_equal_flash_patterns(self):

        entropy_values = []
        for equal_flash_pattern in self.all_equal_flash_patterns:

            index_to_differentiate = np.where(self.hypothesis_scores == 1)[0]

            labels = equal_flash_pattern[index_to_differentiate]

            value, counts = np.unique(labels, return_counts=True)
            entropy = scipy.stats.entropy(counts, base=2)

            entropy_values.append(entropy)

        max_indexes = np.where(entropy_values == np.max(entropy_values))[0]

        return operator.itemgetter(*max_indexes)(self.all_equal_flash_patterns)

    def update(self, flash_pattern, feedback_label):
        self.feedback_history.append((flash_pattern, feedback_label))

        step_scores = flash_pattern == np.array(feedback_label)

        self.hypothesis_scores *= step_scores

    def is_solved(self):
        return np.sum(self.hypothesis_scores) == 1

    def is_inconsistent(self):
        return np.sum(self.hypothesis_scores) == 0
