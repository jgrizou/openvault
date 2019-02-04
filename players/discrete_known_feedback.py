import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


import numpy as np


class DiscreteKnownFeedbackPlayer(object):

    def __init__(self, n_hypothesis):
        self.n_hypothesis = n_hypothesis

        self.target_index = np.random.randint(self.n_hypothesis)


    def get_feedback(self, flash_pattern):
        return flash_pattern[self.target_index]
