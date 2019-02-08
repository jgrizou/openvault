import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


from pprint import pprint
import numpy as np
import random

import tools


from discrete import DiscreteLearner
from discrete import DiscretePlayer
from discrete import DiscreteRunner


N_ABORT_STEPS = np.inf
TARGET_INDEX = None
PLANNING_METHOD = 'even_uncertainty'


def run(seed, code_length, n_hypothesis, n_button_per_label, are_labels_known):

    tools.set_seed(seed, verbose=False)

    player_symbols = {
        True: [str(i) for i in range(n_button_per_label)],
        False: [str(i) for i in range(n_button_per_label, 2*n_button_per_label)]
    }

    known_symbols = {}
    if are_labels_known:
        for label in [True, False]:
            for symbol in player_symbols[label]:
                known_symbols[symbol] = label

    runner = DiscreteRunner(n_hypothesis, player_symbols, known_symbols, TARGET_INDEX)

    result = {
        'steps_per_solution' : [],
        'avg_steps_duration_per_solution': []
    }
    for _ in range(code_length):
        run_info = runner.run_until_solved(PLANNING_METHOD, N_ABORT_STEPS)
        runner.reset_learner(run_info['learner_interpretation'])
        runner.update_target_index(TARGET_INDEX)

        # print('Identified {} from {} in {} steps.'.format(run_info['learner_solution_index'], run_info['player_target_index'], run_info['n_steps']))

        result['steps_per_solution'].append(run_info['n_steps'])
        result['avg_steps_duration_per_solution'].append(run_info['avg_steps_duration'])

    return result

if __name__ == '__main__':

    # SEED = np.random.randint(0, 10000)
    # if SEED is not None:
    #     tools.set_seed(SEED, verbose=False)

    import math
    import itertools
    from tinydb import TinyDB, Query

    DB_FILENAME = os.path.join(HERE_PATH, 'results.json')
    db = TinyDB(DB_FILENAME)
    user = Query()

    N_EXPERIMENTS = 100
    XP_PARAMS = {
        'seed': range(N_EXPERIMENTS),
        'code_length': [1, 2, 3, 4, 5],
        'n_hypothesis': [4, 6, 8, 10, 12],
        'n_button_per_label': [1, 2, 3, 4, 5],
        'are_labels_known': [True, False]
    }

    param_keys, param_values = zip(*XP_PARAMS.items())

    n_experiments = sum(1 for _ in itertools.product(*param_values))
    digits = int(math.log10(n_experiments))+1

    for i_exp, param_combination in enumerate(itertools.product(*param_values)):

        i_exp_corrected = i_exp + 1
        i_str = f"{i_exp_corrected:{digits}}"
        n_exp_str = f"{n_experiments:{digits}}"

        print('{}:{} '.format(i_str , n_exp_str), end='\b')

        xp = {}
        xp['params'] = dict(zip(param_keys, param_combination))

        if db.search(user.params == xp['params']):
            print(' - Skipping')
            continue
        else:
            print(' - Running')
            xp['results'] = run(**xp['params'])
            db.insert(xp)
