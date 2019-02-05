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
        self.hypothesis_interpretation = self.init_hypothesis_interpretation()

        self.all_equal_flash_patterns = self.get_all_equal_flash_patterns()

    def is_solved(self):
        return np.sum(self.hypothesis_scores) == 1

    def get_solution(self):
        if self.is_solved():
            solution_index = self.hypothesis_scores.tolist().index(1)
            solution_interpretation = self.get_interpretation_for_hypothesis_index(solution_index)
            return solution_index, solution_interpretation
        else:
            raise Exception('Not Solved')

    def is_inconsistent(self):
        return np.sum(self.hypothesis_scores) == 0

    def init_hypothesis_interpretation(self):
        init_hypothesis_interpretation = []
        for i in range(self.n_hypothesis):
            init_hypothesis_interpretation.append({})
        return init_hypothesis_interpretation

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

        entropy_values = []
        for equal_flash_pattern in self.all_equal_flash_patterns:
            hypothesis_index_to_differentiate = np.where(self.hypothesis_scores == 1)[0]

            expectation_from_pattern = []
            for i_hyp_diff in hypothesis_index_to_differentiate:
                label = equal_flash_pattern[i_hyp_diff]
                hypothesis_interpretation = self.get_interpretation_for_hypothesis_index(i_hyp_diff)
                reverse_interpretation = self.build_reverse_interpretation(hypothesis_interpretation)

                if label in reverse_interpretation:
                    expectation_from_pattern.append(reverse_interpretation[label])
                else:
                    expectation_from_pattern.append(set([RESERVED_UNKNOWN_SYMBOL]))

            intersect = expectation_from_pattern[0]
            difference = expectation_from_pattern[0]
            for i_set in expectation_from_pattern[1:]:
                intersect = i_set & intersect
                difference = i_set ^ difference
            entropy_values.append(len(difference) - len(intersect))

        max_indexes = np.where(entropy_values == np.max(entropy_values))[0]

        if len(max_indexes) == 1:
            uncertain_flash_patterns = [self.all_equal_flash_patterns[max_indexes[0]]]
        else:
            uncertain_flash_patterns = list(operator.itemgetter(*max_indexes)(self.all_equal_flash_patterns))

        # print(len(uncertain_flash_patterns))

        return uncertain_flash_patterns

    def get_interpretation_for_hypothesis_index(self, hypothesis_index):
        dict_hyp = self.hypothesis_interpretation[hypothesis_index]
        hypothesis_interpretation = {}
        for symbol, interpretation in dict_hyp.items():
            interpretation = np.array(interpretation)
            nan_free_interpretation = interpretation[~np.isnan(interpretation)]
            # because it should be valid, this should be consistent and have only one value
            values = np.unique(nan_free_interpretation)
            if len(values) > 1:
                raise Exception('Hypothesis inconsistent')
            hypothesis_interpretation[symbol] = int(values[0])
        return hypothesis_interpretation

    def build_reverse_interpretation(self, hypothesis_interpretation):
        reverse_interpretation = {}
        for symbol, meaning in hypothesis_interpretation.items():
            if meaning not in reverse_interpretation:
                reverse_interpretation[meaning] = set()
            reverse_interpretation[meaning].add(symbol)
        return reverse_interpretation

    def update(self, flash_pattern, feedback_symbol):

        if feedback_symbol == RESERVED_UNKNOWN_SYMBOL :
            raise Exception('{} is a reserved internal symbol'.format(feedback_symbol))

        self.flash_history.append(flash_pattern)
        self.feedback_history.append(feedback_symbol)

        n_iteration = len(self.feedback_history)

        # compute interpretations
        for n_hyp in range(self.n_hypothesis):
            tmp_interpreation = flash_pattern[n_hyp]

            dict_hyp = self.hypothesis_interpretation[n_hyp]

            if feedback_symbol not in dict_hyp:
                # print('!! UNSEEN {}'.format(feedback_symbol))
                dict_hyp[feedback_symbol] = np.full(n_iteration-1, np.nan).tolist()

            for symbol, interpretation in dict_hyp.items():
                if symbol == feedback_symbol:
                    interpretation.append(tmp_interpreation)
                else:
                    interpretation.append(np.nan)

            self.hypothesis_interpretation[n_hyp] = dict_hyp

        #compute scores by measuring consistency of interpretation
        for n_hyp in range(self.n_hypothesis):

            consistent = True

            dict_hyp = self.hypothesis_interpretation[n_hyp]
            for symbol, interpretation in dict_hyp.items():
                interpretation = np.array(interpretation)
                nan_free_interpretation = interpretation[~np.isnan(interpretation)]

                # we should compute entropy but this is faster as we only care about consistency 1 symbol -> 1 meaning
                values = np.unique(nan_free_interpretation)
                if len(values) > 1:
                    consistent = False

            if consistent:
                self.hypothesis_scores[n_hyp] = 1
            else:
                self.hypothesis_scores[n_hyp] = 0


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
