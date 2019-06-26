import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding openvault directory to path for script ot work both if called directly or imported
import sys
openvault_path = os.path.join(HERE_PATH, '..')
sys.path.append(openvault_path)


import time
import random
import itertools

import numpy as np

from openvault import tools

RESERVED_UNKNOWN_SYMBOL = 'UNKNOWN'
INCONSISTENT_SYMBOL = 'INCONSISTENT'


class DiscretePlayer(object):

    def __init__(self, n_hypothesis, player_symbols, target_index=None):
        self.n_hypothesis = n_hypothesis
        self.player_symbols = player_symbols
        self.update_target_index(target_index)

    def get_feedback_symbol(self, flash_pattern):
        is_target_flashed = flash_pattern[self.target_index]
        associated_symbols = self.player_symbols[is_target_flashed]
        return random.choice(associated_symbols)

    def update_target_index(self, new_target_index):
        if new_target_index is None:  # set it randomly
            self.target_index = np.random.randint(self.n_hypothesis)
        else:
            self.target_index = new_target_index


class DiscreteLearner(object):

    def __init__(self, n_hypothesis, known_symbols={}):
        self.n_hypothesis = n_hypothesis
        self.known_symbols = known_symbols

        self.hypothesis_validity = [True for _ in range(n_hypothesis)]

        self.hypothesis_validity_history = []
        self.hypothesis_validity_history.append(self.hypothesis_validity.copy())

        self.flash_history = []
        self.symbol_history = []

        self.hypothesis_labels = [[] for _ in range(n_hypothesis)]

        self.even_flash_patterns = tools.compute_even_flash_patterns(n_hypothesis)

    def is_inconsistent(self):
        return np.sum(self.hypothesis_validity) == 0

    def is_solved(self):
        return np.sum(self.hypothesis_validity) == 1

    def get_solution_index(self):
        if self.is_solved():
            return self.hypothesis_validity.index(True)
        else:
            raise Exception('Not Solved yet')

    def compute_symbols_belief_for_hypothesis(self, hypothesis_index):
        symbol_set = set(self.symbol_history)
        hyp_labels = self.hypothesis_labels[hypothesis_index]

        symbols_belief = self.known_symbols.copy()
        for symbol in symbol_set:
            symbol_index = tools.get_indexes_for_value(self.symbol_history, symbol)
            labels_for_symbol = tools.get_values_at_indexes(hyp_labels, symbol_index)

            associated_labels = set(labels_for_symbol)

            # len associated_labels cannot be 0
            if len(associated_labels) == 0:
                raise Exception('Problem in code just above, symbol in set not found in history!')

            if len(associated_labels) == 1:
                label_belief = list(associated_labels)[0]
                if symbol not in symbols_belief:
                    symbols_belief[symbol] = label_belief
                elif label_belief != symbols_belief[symbol]:
                    # Symbol belief inconsistent between known and observed!
                    symbols_belief[symbol] = INCONSISTENT_SYMBOL
            else:
                # Symbol belief inconsistent between known and observed!
                symbols_belief[symbol] = INCONSISTENT_SYMBOL

        return symbols_belief

    def update(self, flash_pattern, feedback_symbol):

        if feedback_symbol == RESERVED_UNKNOWN_SYMBOL :
            raise Exception('{} is a reserved internal symbol!'.format(feedback_symbol))

        # log data
        self.flash_history.append(flash_pattern)
        self.symbol_history.append(feedback_symbol)

        # update interpretations according to each hypothesis
        for i_hyp in range(self.n_hypothesis):
            self.hypothesis_labels[i_hyp].append(flash_pattern[i_hyp])

        # compute validity of each hypothesis by measuring either:
        # - what the user meant using known symbols
        # - use the consistency of interpretation

        # only do so for hypothesis that are still valid
        valid_hyp_index = tools.get_indexes_for_value(self.hypothesis_validity, True)
        for i_hyp in valid_hyp_index:

            # if latest symbol is known, update directly the validity
            if feedback_symbol in self.known_symbols:
                feedback_label = self.known_symbols[feedback_symbol]
                flashed_label = flash_pattern[i_hyp]
                # a feedback is consistent with the hypothesis if it match with the flash.
                # If flash -> symbol for flash.
                #If no flash -> symbol for no flash
                consistent = feedback_label == flashed_label

            else: # compute consistency
                # list all unique symbols observed so far
                symbols_set = set(self.symbol_history)

                consistent = True
                for symbol in symbols_set:
                    # for each symbol find what labels it refers to
                    symbol_indexes = tools.get_indexes_for_value(self.symbol_history, symbol)
                    hyp_labels = self.hypothesis_labels[i_hyp]
                    labels_of_symbol = tools.get_values_at_indexes(hyp_labels, symbol_indexes)

                    # check the conistency of the use of the symbol
                    # we define consistency as the use of one symbol for only one meaning only
                    if len(set(labels_of_symbol)) > 1:
                        consistent = False
                        # we can break now as we found an inconsistency, no need to check more for this i_hyp
                        break

            # assign value
            self.hypothesis_validity[i_hyp] = consistent

        # logging the updated hypothesis_validity
        self.hypothesis_validity_history.append(self.hypothesis_validity.copy())


    def get_next_flash_pattern(self, planning_method='even_uncertainty'):
        if planning_method == 'random':
            return [random.choice([True, False]) for _ in range(self.n_hypothesis)]
        elif planning_method == 'even_random':
            return random.choice(self.even_flash_patterns)
        elif planning_method == 'even_uncertainty':
            return random.choice(self.compute_uncertain_patterns())
        else:
            raise Exception('Planning method "{}" not defined'.format(method))


    def compute_uncertain_patterns(self):

        symbol_set = set(self.symbol_history)
        symbol_set.add(RESERVED_UNKNOWN_SYMBOL) # we add an unknown symbol here for those cases when we have not seen any symbol for label yet

        # we will compute an uncertainty score for each pattern
        pattern_scores = []
        for even_flash_pattern in self.even_flash_patterns:
            # we only compute uncertainty between hypothesis still valid
            hypothesis_indexes_to_differentiate = tools.get_indexes_for_value(self.hypothesis_validity, True)

            # for each hypothesis, we empirically compute the distribution of expected symbols from past observation
            expected_symbol_distributions = []
            for i_hyp in hypothesis_indexes_to_differentiate:
                # for each hyppthesis, find the label expected
                label = even_flash_pattern[i_hyp]
                # get the history of labels for this hypothesis
                hyp_labels = self.hypothesis_labels[i_hyp]
                # find the index of each iteration we observed that label
                label_indexes = tools.get_indexes_for_value(hyp_labels, label)
                # find the symbols associated
                expected_symbols = tools.get_values_at_indexes(self.symbol_history, label_indexes)

                # if empty, we have not seen that yet, we add the placeholder for an unknown symbol
                if len(expected_symbols) == 0:
                    expected_symbols.append(RESERVED_UNKNOWN_SYMBOL)

                # now we compute the distribution of the previously observed symbols
                # we prepare an empty distribution of all observed symbols so far
                symbols_distribution = [0 for _ in range(len(symbol_set))]
                # for each observed symbol, count the number of times we observed it
                for i, symbol in enumerate(symbol_set):
                    symbols_distribution[i] = expected_symbols.count(symbol)

                # we normalize the distribution
                symbols_distribution /= np.sum(symbols_distribution)
                # and we add it to the list
                expected_symbol_distributions.append(symbols_distribution)

            # we now have a full distribution of expected symbols
            # we compute an uncertainty score by measruing the pairwise probability that two hypothesis will trigger different feedback symbol from the user
            # in other words we measure how uncertain we are of the outcome of the tested flashing pattern
            uncertainty_score = 0
            pairwise_combination_indexes = itertools.combinations(range(len(expected_symbol_distributions)), 2)
            for i, j in pairwise_combination_indexes:
                i_dist = np.array(expected_symbol_distributions[i])
                j_dist = np.array(expected_symbol_distributions[j])
                # 1 minus the joint probablity
                uncertainty = 1 - np.dot(i_dist, j_dist)
                # we simply sum the scores over all pairs
                uncertainty_score += uncertainty

            ## add to pattern_scores
            pattern_scores .append(uncertainty_score)

        # we have a score for every patterns now
        # we return an array of the most uncertain one
        max_scores = np.max(pattern_scores)
        max_scores_indexes = tools.get_indexes_for_value(pattern_scores, max_scores)
        best_flash_patterns = tools.get_values_at_indexes(self.even_flash_patterns, max_scores_indexes)

        return best_flash_patterns

    def get_logs(self):

        logs = {}

        logs['learner_type'] = 'discrete'

        logs['n_hypothesis'] = self.n_hypothesis
        logs['known_symbols'] = self.known_symbols

        logs['hypothesis_validity_history'] = self.hypothesis_validity_history

        logs['flash_history'] = self.flash_history
        logs['symbol_history'] = self.symbol_history

        logs['hypothesis_labels'] = self.hypothesis_labels

        logs['is_inconsistent'] = bool(self.is_inconsistent())  # bool(for json serialisation)
        logs['is_solved'] = bool(self.is_solved())
        if self.is_solved():
            logs['solution_index'] = int(self.get_solution_index())

        symbols_belief_per_hypothesis = []
        for i in range(self.n_hypothesis):
            symbol_belief = self.compute_symbols_belief_for_hypothesis(i)
            symbols_belief_per_hypothesis.append(symbol_belief)
        logs['symbols_belief_per_hypothesis'] = symbols_belief_per_hypothesis

        return logs
