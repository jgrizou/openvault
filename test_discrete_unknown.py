import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import time

import numpy as np

from discrete_unknown_2 import DiscreteUnknownFeedbackLearner
from discrete_unknown_2 import DiscreteUnknownFeedbackPlayer

# from discrete_unknown import DiscreteUnknownFeedbackLearner
# from discrete_unknown import DiscreteUnknownFeedbackPlayer

if __name__ == '__main__':

    N_HYPOTHESIS = 10
    TRUE_SUMBOLS = ['yes', 'y', 'oui']
    FALSE_SUMBOLS= ['no', 'n', 'non']
    # TRUE_SUMBOLS = ['yes']
    # FALSE_SUMBOLS= ['no']
    ABORT_STEPS = 20


    steps_to_solution = []
    for _ in range(50):

        learner = DiscreteUnknownFeedbackLearner(N_HYPOTHESIS)
        player = DiscreteUnknownFeedbackPlayer(N_HYPOTHESIS, TRUE_SUMBOLS, FALSE_SUMBOLS)

        # print('Target is index: {}'.format(player.target_index))

        cnt = 0
        while not learner.is_solved():
            start_time = time.time()
            cnt+=1

            previous_scores = learner.hypothesis_scores

            # print('###')

            flash_pattern = learner.get_next_flash_pattern()
            # feedback_label = np.array(int(input('Enter Feedback: ')))
            feedback_label = player.get_feedback(flash_pattern)
            learner.update(flash_pattern, feedback_label)

            # print('Previous scores:  {}'.format(previous_scores))
            # print('Pattern: {}'.format(flash_pattern))
            # print('Feedback: {}'.format(feedback_label))
            # print('Scores:  {}'.format(learner.hypothesis_scores))

            if learner.is_inconsistent():
                raise Exception('Learner Inconsistent!')

            elapsed = time.time() - start_time
            # print('Step time: {}'.format(elapsed))

            if cnt > ABORT_STEPS:
                break


        steps_to_solution.append(cnt)


        # print('###')

        if learner.is_solved():
            print('Found solution in {} steps'.format(cnt))

            # solution_index, solution_interpreation = learner.get_solution()
            # print('{} : {}'.format(player.target_index, solution_index))
            # print('{}'.format(solution_interpreation))
        else:
            print('Aborted after {} iterations'.format(ABORT_STEPS))

        # import pprint
        # pprint.pprint(learner.hypothesis_interpretation)

    print(np.mean(steps_to_solution))
    print(np.std(steps_to_solution))
