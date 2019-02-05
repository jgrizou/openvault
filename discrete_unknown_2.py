import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import pprint
import random
import operator
import itertools

import scipy.stats
import numpy as np

RESERVED_UNKNOWN_SYMBOL = 'UNKNOWN'

class DiscreteUnknownFeedbackLearner(object):

    def __init__(self, n_hypothesis):
        self.n_hypothesis = n_hypothesis

        self.hypothesis_scores = np.ones(self.n_hypothesis, dtype=np.int)

        self.flash_history = []
        self.feedback_history = []
        self.hypothesis_labels = [[] for _ in range(self.n_hypothesis)]

        self.all_equal_flash_patterns = self.get_all_equal_flash_patterns()

    def is_solved(self):
        return np.sum(self.hypothesis_scores) == 1

    def is_inconsistent(self):
        return np.sum(self.hypothesis_scores) == 0

    def get_all_equal_flash_patterns(self):
        all_equal_flash_patterns = []
        for comb in itertools.combinations(range(self.n_hypothesis), self.n_hypothesis//2):
            flash_pattern = np.zeros(self.n_hypothesis, dtype=np.int)
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

        symbols_set = set(self.feedback_history)
        symbols_set.add(RESERVED_UNKNOWN_SYMBOL)

        symbols_set_list = list(symbols_set)

        entropy_values = []
        for equal_flash_pattern in self.all_equal_flash_patterns:
            hypothesis_index_to_differentiate = np.where(self.hypothesis_scores == 1)[0]

            # print('--')
            # print(equal_flash_pattern)

            expected_symbol_distributions = []
            for i_hyp_diff in hypothesis_index_to_differentiate:
                label = equal_flash_pattern[i_hyp_diff]
                label_index = np.where(self.hypothesis_labels[i_hyp_diff] == label)[0]

                if len(label_index) > 0:
                    if len(label_index) == 1:
                        symbols_of_label = [self.feedback_history[label_index[0]]]
                    else:
                        symbols_of_label = list(operator.itemgetter(*label_index)(self.feedback_history))
                else:
                    symbols_of_label = [RESERVED_UNKNOWN_SYMBOL]

                ##
                symbols_distribution = [[] for _ in range(len(symbols_set_list))]
                for i, symbol in enumerate(symbols_set_list):
                    symbols_distribution[i] = symbols_of_label.count(symbol)

                symbols_distribution /= np.sum(symbols_distribution)

                expected_symbol_distributions.append(symbols_distribution)

            # print(expected_symbol_distributions)
            # print(hypothesis_index_to_differentiate)

            scores = []
            for i, j in itertools.combinations(range(len(expected_symbol_distributions)), 2):
                i_dist = np.array(expected_symbol_distributions[i])
                j_dist = np.array(expected_symbol_distributions[j])
                score = 1 - np.dot(i_dist, j_dist)
                scores.append(score)

            # print(scores)
            # print(np.sum(scores))

            entropy_values.append(np.sum(scores))

        max_indexes = np.where(entropy_values == np.max(entropy_values))[0]

        if len(max_indexes) == 1:
            uncertain_flash_patterns = [self.all_equal_flash_patterns[max_indexes[0]]]
        else:
            uncertain_flash_patterns = list(operator.itemgetter(*max_indexes)(self.all_equal_flash_patterns))

        # print(len(uncertain_flash_patterns))

        return uncertain_flash_patterns

    def update(self, flash_pattern, feedback_symbol):

        if feedback_symbol == RESERVED_UNKNOWN_SYMBOL :
            raise Exception('{} is a reserved internal symbol'.format(feedback_symbol))

        self.flash_history.append(flash_pattern)
        self.feedback_history.append(feedback_symbol)

        n_iteration = len(self.feedback_history)

        # update interpretations
        for i_hyp in range(self.n_hypothesis):
            self.hypothesis_labels[i_hyp].append(flash_pattern[i_hyp])

        symbols_set = set(self.feedback_history)

        #compute scores by measuring consistency of interpretation
        for i_hyp in range(self.n_hypothesis):

            consistent = True

            for symbol in symbols_set:
                symbol_index = np.where(np.array(self.feedback_history) == symbol)[0]
                labels = self.hypothesis_labels[i_hyp]
                if len(symbol_index) == 1:
                    labels_of_symbol = [labels[symbol_index[0]]]
                else:
                    labels_of_symbol = list(operator.itemgetter(*symbol_index)(labels))

                # we should compute entropy but this is faster as we only care about consistency 1 symbol -> 1 meaning
                values = np.unique(labels_of_symbol)
                if len(values) > 1:
                    consistent = False

            if consistent:
                self.hypothesis_scores[i_hyp] = 1
            else:
                self.hypothesis_scores[i_hyp] = 0


class DiscreteUnknownFeedbackPlayer(object):

    def __init__(self, n_hypothesis, true_symbols=['y'], false_symbol=['n'], target_index=None):
        self.n_hypothesis = n_hypothesis

        if target_index is not None:
            self.target_index = target_index
        else:
            self.target_index = np.random.randint(self.n_hypothesis)

        self.true_symbols = true_symbols
        self.false_symbol = false_symbol


    def get_feedback(self, flash_pattern):
        if flash_pattern[self.target_index]:
            return random.choice(self.true_symbols)
        else:
            return random.choice(self.false_symbol)
