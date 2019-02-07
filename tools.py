import random
import itertools

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
        even_flash_patterns.append(flash_pattern)
    return even_flash_patterns

def get_values_at_indexes(a_list, indexes):
    return [a_list[i] for i in indexes]

def get_indexes_for_value(a_list, value):
    return [i for i, x in enumerate(a_list) if x == value]
