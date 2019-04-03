
import numpy as np
from sklearn.svm import SVC


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class ContinuousPlayer(object):

    def __init__(self, n_hypothesis, signal_generator, target_index=None):
        self.n_hypothesis = n_hypothesis
        self.signal_generator = signal_generator
        self.update_target_index(target_index)

    def get_feedback_signal(self, flash_pattern):
        is_target_flashed = flash_pattern[self.target_index]
        return self.signal_generator(is_target_flashed)

    def update_target_index(self, new_target_index):
        if new_target_index is None:  # set it randomly
            self.target_index = np.random.randint(self.n_hypothesis)
        else:
            self.target_index = new_target_index


if __name__ == '__main__':


    plt.ion()

    from tools import generator_2D

    player = ContinuousPlayer(2, generator_2D, 0)

    X = []
    y = []
    for i in range(20):
        X.append(player.get_feedback_signal([0,1]))
        y.append(0)
        X.append(player.get_feedback_signal([1,0]))
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    # plt.figure()
    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.show()


    from tools import compute_loglikelihood
    from tools import is_y_valid


    print(is_y_valid(y))
    l1 = compute_loglikelihood(X, y, 'rbf')

    np.random.shuffle(y)
    print(is_y_valid(y))
    l2 = compute_loglikelihood(X, y, 'rbf')

    np.random.shuffle(y)
    print(is_y_valid(y))
    l3 = compute_loglikelihood(X, y, 'rbf')

    from log_tools import normalize_log_array

    # ideally we should do leaveoneout predictions
    # and use some sort of statistical analysis instead of comparing loglikelihood, but works and is fast


    lnorm = normalize_log_array([[l1, l2, l3]])


    # plt.figure()
    # plt.scatter(X[:,0], X[:,1], c=np.exp(a[:,0]))
    # plt.show()
