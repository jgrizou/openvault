import numpy as np


def label_log_proba(y, classes, proba_label_valid=0.99):

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

    return logproba_label

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
