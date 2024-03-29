import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding openvault directory to path for script ot work both if called directly or imported
import sys
openvault_path = os.path.join(HERE_PATH, '..')
sys.path.append(openvault_path)

from openvault import tools

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.calibration import CalibratedClassifierCV


def train_classifier(X, y, kernel):

    clf = SVC(gamma='scale', kernel=kernel, probability=False)
    clf.fit(X, y)

    # we need to use a calibrator to get consistent probability matrix output
    # see https://github.com/scikit-learn/scikit-learn/issues/13211#issuecomment-511392497
    calibrator = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrator.fit(X, y)

    # option with cross validation for calibration
    # the resulting maps are not that nice for the user to see on the interface of openvault_web, so decided not to use it
    # clf = SVC(gamma='scale', kernel=kernel, probability=True)
    # cv = np.min([get_min_sample_per_class(y), 10])
    # calibrator = CalibratedClassifierCV(clf, method='sigmoid', cv=cv)
    # calibrator.fit(X, y)

    return calibrator


def compute_loglikelihood(X, y, kernel='rbf', proba_label_valid=0.99, use_leave_one_out=True, seed_value=0):

    X = np.atleast_2d(X)
    y = np.array(y)

    if seed_value is not None:
        # we force the random seed here to ensure a classifier trained on the same data will lead to the same final decision function
        # not necessary but needed for a nice visiualisation of the underlying process by user on the web interface
        # set seed_value to none to not reset seed at this point
        tools.set_seed(seed_value, verbose=False)
    # fit a classifier on the full data anyway as we need one for planning (etc) that this function will return
    clf = train_classifier(X, y, kernel)

    # this is valid only because we use a calibrator in train_classifier
    # otherwise very annoying problems can happen as the classes_ variable does not always reflects the ordering of column in the proba prediction matrice, see: https://github.com/scikit-learn/scikit-learn/issues/13211#issuecomment-511392497
    ordered_classes = clf.classes_

    # get true log matrice
    log_y_true = label_log_proba(y, ordered_classes, proba_label_valid)

    # compute predicted log matrice
    if use_leave_one_out:
        log_y_pred = np.zeros(log_y_true.shape)

        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):

            loo_clf = train_classifier(X[train_index], y[train_index], kernel)
            log_y_test_pred = np.log(loo_clf.predict_proba(X[test_index]))
            log_y_pred[test_index, :] = log_y_test_pred

    else:
        log_y_pred = np.log(clf.predict_proba(X))

    # likelihood that the classifier output matches with the labels:
    # prod_i sum_y P(y_true_{i}=y)P(y_pred_{i}=y)
    # but done all in log to avoid numerical issues
    loglikelihood = np.sum(sum_log_array(log_y_true + log_y_pred))

    classifier_info = {}
    classifier_info['clf'] = clf
    classifier_info['ordered_classes'] = ordered_classes
    classifier_info['log_y_true'] = log_y_true
    classifier_info['log_y_pred'] = log_y_pred

    return loglikelihood, classifier_info


def label_log_proba(y, classes, proba_label_valid):

    N_CLASSES = classes.shape[0]

    logproba_label = []
    for label in y:
        proba_label = []
        for c in classes:
            if label == c:
                proba = proba_label_valid
            else:
                proba = (1 - proba_label_valid) / (N_CLASSES - 1)
            proba_label.append(proba)

        # make it log, and disable warning for divide by zero that happen when we do np.log(0)
        np.seterr(divide = 'ignore')
        logproba_label.append(np.log(proba_label))
        np.seterr(divide = 'warn')

    return np.array(logproba_label)


def add_lns(a_ln, b_ln):
    # return P(a)+P(b) but directly from ln(a) and ln(b).

    # if both a_ln and b_ln are equal to -inf, then we return -inf
    if np.equal(a_ln, -np.inf) and np.equal(b_ln, -np.inf):
        return -np.inf

    # if a_ln is -inf, it mean P(a)=0, so we return b_ln
    if np.equal(a_ln, -np.inf):
        return b_ln

    # if b_ln is -inf, it mean P(b)=0, so we return a_ln
    if np.equal(b_ln, -np.inf):
        return a_ln

    # Else we do the proper calculation
    # The method is based on the notion that
    # ln(a + b) = ln{exp[ln(a) - ln(b)] + 1} + ln(b).
    return np.log(np.exp(a_ln - b_ln) + 1) + b_ln


def sum_log_array(log_array):
    log_array = np.atleast_2d(log_array)
    (n_lines, n_columns) = log_array.shape

    out = np.zeros(shape=(n_lines, ))
    for i in range(n_lines):
        if n_columns > 1:
            logSum = add_lns(log_array[i, 0], log_array[i, 1])
            for j in range(n_columns - 2):
                logSum = add_lns(logSum, log_array[i, j + 2])
        else:
            logSum = log_array[i, 0]
        out[i] = logSum

    return out

def normalize_log_array(log_array):
    init_shape = np.array(log_array).shape

    log_array = np.atleast_2d(log_array)
    (n_lines, n_columns) = log_array.shape

    logsum_array = sum_log_array(log_array)

    out = np.zeros(shape=(n_lines, n_columns))
    for i in range(n_lines):
        out[i, :] = np.exp(log_array[i, :] - logsum_array[i])

    return np.reshape(out, init_shape)


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


def weighted_variance(values, weights):
    weighted_average = np.average(values, weights=weights)
    weighted_variance = np.average((values-weighted_average)**2, weights=weights)
    return weighted_variance


def reorder_column_according_to_target_classes_ordering(X, current_classes, target_classes):

    if np.all(current_classes == target_classes):
        return X
    else:
        raise Execption('Not implemented, should never happen at this stage if a CalibratedClassifierCV is used')
