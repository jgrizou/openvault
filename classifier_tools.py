import numpy as np
from sklearn.svm import SVC


def compute_loglikelihood(X, y, kernel='rbf', proba_label_valid=0.99):

    X = np.atleast_2d(X)
    y = np.array(y)

    clf = SVC(gamma='scale', kernel=kernel, probability=True)
    clf.fit(X, y)

    log_y_true = label_log_proba(y, clf.classes_, proba_label_valid)
    log_y_pred = clf.predict_log_proba(X)

    # likelihood that the classifier output matches with the labels:
    # prod_i sum_y P(y_true_{i}=y)P(y_pred_{i}=y)
    # but done all in log to avoid numerical issues
    loglikelihood = np.sum(sum_log_array(log_y_true + log_y_pred))

    classifier_info = {}
    classifier_info['clf'] = clf
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
        logproba_label.append(np.log(proba_label))

    return np.array(logproba_label)

# The method is based on the notion that
# ln(a + b) = ln{exp[ln(a) - ln(b)] + 1} + ln(b).
def add_lns(a_ln, b_ln):
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
            logSum = log_array[i, 1]
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
