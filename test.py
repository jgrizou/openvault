

from pprint import pprint
import numpy as np

import tools



from discrete import DiscreteLearner
from discrete import DiscretePlayer



if __name__ == '__main__':

    SEED = np.random.randint(0, 1000)
    N_ABORT_STEPS = 20

    if SEED is not None:
        tools.set_seed(SEED)

    N_HYPOTHESIS = 10

    KNOWN_SYMBOLS = {
        'ouioui': True,
        'nonnon': False
    }
    KNOWN_SYMBOLS = {}

    learner = DiscreteLearner(N_HYPOTHESIS, KNOWN_SYMBOLS)


    PLAYER_SYMBOLS = {
        True: ['yes', 'y', 'oui', 'e'],
        False: ['no', 'n', 'non']
    }
    # PLAYER_SYMBOLS = {
    #     True: ['yes'],
    #     False: ['no']
    # }
    TARGET_INDEX = None

    player = DiscretePlayer(N_HYPOTHESIS, PLAYER_SYMBOLS, TARGET_INDEX)

    print('Target is {}'.format(player.target_index))

    cnt = 0
    while not learner.is_solved():
        cnt+=1

        previous_scores = learner.hypothesis_validity.copy()

        # print('###')

        flash_pattern = learner.get_next_flash_pattern()
        # feedback_label = np.array(int(input('Enter Feedback: ')))
        feedback_symbol = player.get_feedback_symbol(flash_pattern)
        learner.update(flash_pattern, feedback_symbol)

        # print('###')
        # print('Previous Valid:  {}'.format(previous_scores))
        # print('Pattern: {}'.format(flash_pattern))
        # print('Feedback: {}'.format(feedback_symbol))
        # print('Validity:  {}'.format(learner.hypothesis_validity))

        if learner.is_inconsistent():
            raise Exception('Learner Inconsistent!')

        if cnt == N_ABORT_STEPS:
            break

    if learner.is_solved():
        print('Found solution in {} steps'.format(cnt))

        solution_index = learner.get_solution_index()
        print('Estimated Target is {}'.format(solution_index))

        solution_interpretation = learner.compute_symbols_belief_for_hypothesis(solution_index)
        pprint(solution_interpretation)

    else:
        print('Aborted after {} iterations'.format(cnt))



    print('#################################')

    KNOWN_SYMBOLS = solution_interpretation

    learner = DiscreteLearner(N_HYPOTHESIS, KNOWN_SYMBOLS)

    PLAYER_SYMBOLS = {
        True: ['yes', 'y', 'oui', 'e'],
        False: ['no', 'n', 'non', 'ewf']
    }
    TARGET_INDEX = None

    player = DiscretePlayer(N_HYPOTHESIS, PLAYER_SYMBOLS, TARGET_INDEX)

    print('Target is {}'.format(player.target_index))

    cnt = 0
    while not learner.is_solved():
        cnt+=1

        previous_scores = learner.hypothesis_validity.copy()

        # print('###')

        flash_pattern = learner.get_next_flash_pattern()
        # feedback_label = np.array(int(input('Enter Feedback: ')))
        feedback_symbol = player.get_feedback_symbol(flash_pattern)
        learner.update(flash_pattern, feedback_symbol)

        # print('###')
        # print('Previous Valid:  {}'.format(previous_scores))
        # print('Pattern: {}'.format(flash_pattern))
        # print('Feedback: {}'.format(feedback_symbol))
        # print('Validity:  {}'.format(learner.hypothesis_validity))

        if learner.is_inconsistent():
            raise Exception('Learner Inconsistent!')

        if cnt == N_ABORT_STEPS:
            break

    if learner.is_solved():
        print('Found solution in {} steps'.format(cnt))

        solution_index = learner.get_solution_index()
        print('Estimated Target is {}'.format(solution_index))

        solution_interpretation = learner.compute_symbols_belief_for_hypothesis(solution_index)
        pprint(solution_interpretation)

    else:
        print('Aborted after {} iterations'.format(cnt))
