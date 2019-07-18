import random
import itertools

import scipy.stats
import numpy as np


def set_seed(seed, verbose=True):
    if verbose:
        print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)


def compute_even_flash_patterns(n_hypothesis):
    even_flash_patterns = []
    for combination in itertools.combinations(range(n_hypothesis), n_hypothesis//2):
        flash_pattern = np.full(n_hypothesis, False)
        flash_pattern[np.array(combination)] = True
        even_flash_patterns.append(flash_pattern.tolist())
    return even_flash_patterns


def get_values_at_indexes(a_list, indexes):
    return [a_list[i] for i in indexes]


def get_indexes_for_value(a_list, value):
    return [i for i, x in enumerate(a_list) if x == value]


def entropy(labels, base=2):
  value, counts = np.unique(labels, return_counts=True)
  return scipy.stats.entropy(counts, base=base)


def select_high_entropy_patterns(n_hypothesis, hypothesis_labels, flash_patterns):
    """
    We return the flash_patterns that will increase the most (or decrease the less) the entropy on the label for each hypothesis

    This means we try to keep a roughly equal proportion of point belonging to each class in each hypothesis labelling

    This should help get the user less bored by always asking him the same color which might happen sometime based on the uncertainty planning
    """
    pattern_scores = []
    for flash_pattern in flash_patterns:

        score = 0
        for i_hyp in range(n_hypothesis):

            # current entropy of labels
            current_labels = hypothesis_labels[i_hyp].copy()             # we need to copy to not temper with the array
            label_entropy_start = entropy(current_labels)

            # future entropy is the flash pattern is shown
            future_added_label = flash_pattern[i_hyp]
            current_labels.append(future_added_label)
            label_entropy_end = entropy(current_labels)

            # diff entropy is added to score
            # our policy is to increase label entropy to have a roughly equal number of labels in each class for each hypothesis
            label_entropy_diff = label_entropy_end - label_entropy_start
            score += label_entropy_diff

        pattern_scores.append(score)

    # we have a score for every patterns now
    # we return an array of the one with best scores
    max_scores = np.max(pattern_scores)
    max_scores_indexes = get_indexes_for_value(pattern_scores, max_scores)
    best_flash_patterns = get_values_at_indexes(flash_patterns, max_scores_indexes)

    return best_flash_patterns


def remove_patterns_from_list(original_flash_patterns_list, flash_patterns_to_remove):
    # copy list to avoid any out of scope modification
    updated_flash_patterns_list = original_flash_patterns_list.copy()
    for flash_pattern in flash_patterns_to_remove:
        updated_flash_patterns_list.remove(flash_pattern)

    return updated_flash_patterns_list
