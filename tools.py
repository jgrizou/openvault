import random
import itertools

import numpy as np
from sklearn.svm import SVC


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


##
from log_tools import label_log_proba
from log_tools import sum_log_array


def get_min_sample_per_class(y):
    min_sample_per_class = np.inf
    for class_number in np.unique(y):
        n_sample = np.sum(y == class_number)
        if n_sample < min_sample_per_class:
            min_sample_per_class = n_sample
    return min_sample_per_class


def is_y_valid(y, n_class=2, min_sample_per_class=2):
    if len(np.unique(y)) == n_class:
        return get_min_sample_per_class(y) >= min_sample_per_class
    else:
        return False


def generator_2D(is_target_flashed):
    cov = [[0.1, 0], [0, 0.1]]
    if is_target_flashed:
        mean = [0, 0]
    else:
        mean = [1, 1]
    return np.random.multivariate_normal(mean, cov, 1)[0]


def compute_loglikelihood(X, y, kernel='rbf'):

    X = np.atleast_2d(X)
    y = np.array(y)

    clf = SVC(gamma='scale', kernel=kernel, probability=True)
    clf.fit(X, y)

    log_y_true = label_log_proba(y, clf.classes_)
    log_y_pred = clf.predict_log_proba(X)

    # likelihood that the classifier output matches with the labels: prod_i sum_y P(y_true_{i}=y)P(y_pred_{i}=y) but done all in log to avoid numerical issues
    loglikelihood = np.sum(sum_log_array(log_y_true + log_y_pred))

    return loglikelihood
