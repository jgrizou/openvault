import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import numpy as np

from discrete_known import DiscreteKnownFeedbackLearner
from discrete_known import DiscreteKnownFeedbackPlayer

if __name__ == '__main__':

    N_HYPOTHESIS = 10

    learner = DiscreteKnownFeedbackLearner(N_HYPOTHESIS)
    player = DiscreteKnownFeedbackPlayer(N_HYPOTHESIS)


    print('Target is index: {}'.format(player.target_index))

    cnt = 0
    while not learner.is_solved():
        cnt+=1

        print('###')

        flash_pattern = learner.get_next_flash_pattern()
        print('Pattern: {}'.format(flash_pattern))

        # feedback_label = np.array(int(input('Enter Feedback: ')))
        feedback_label = player.get_feedback(flash_pattern)
        print('Feedback: {}'.format(feedback_label))

        learner.update(flash_pattern, feedback_label)
        print('Scores:  {}'.format(learner.hypothesis_scores))

        if learner.is_inconsistent():
            raise Exception('Learner Inconsistent!')

    print('###')

    print('Found solution in {} steps'.format(cnt))
