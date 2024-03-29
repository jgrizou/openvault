import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding openvault directory to path for script ot work both if called directly or imported
import sys
openvault_path = os.path.join(HERE_PATH, '..')
sys.path.append(openvault_path)

import random

import numpy as np

from openvault import tools
from openvault import classifier_tools

N_CLASS_REQUIREMENT = 2  # Feedback mode -> True or False
MIN_SAMPLE_PER_CLASS_REQUIREMENT = 3 # Only compute classifiers when there is enough data for it to be meaningfull. This really depends on the dimensionality of the data, make this a variable in the code if required
MAX_N_SAMPLES_TO_COMPUTE_UNCERTAINTY = 10  # we use the second method (see thesis) to compute uncertainty by sampling previous signals. We limit the number of signals to cap the computational cost.


class ContinuousPlayer(object):

    def __init__(self, n_hypothesis, signal_generator, target_index=None):
        self.n_hypothesis = n_hypothesis
        self.signal_generator = signal_generator
        self.update_target_index(target_index)

    def get_feedback_signal(self, flash_pattern):
        is_target_flashed = flash_pattern[self.target_index]
        return self.signal_generator(is_target_flashed)

    def update_target_index(self, new_target_index=None):
        if new_target_index is None:  # set it randomly
            self.target_index = np.random.randint(self.n_hypothesis)
        else:
            self.target_index = new_target_index


class ContinuousLearner(object):

    def __init__(self, n_hypothesis, proba_decision_threshold=0.9,
        proba_assigned_to_label_valid=1, use_leave_one_out_at_start=True):

        self.n_hypothesis = n_hypothesis

        self.use_leave_one_out = use_leave_one_out_at_start
        self.proba_decision_threshold = proba_decision_threshold # once on hypothesis has proba above this, we consider the problem solved
        self.proba_assigned_to_label_valid = proba_assigned_to_label_valid # 1-P(user is making a mistake) in [0, 1]

        self.hypothesis_probability = [0.5 for _ in range(self.n_hypothesis)]  # this is not really a probability distribution, see thesis eq 4.10
        self.hypothesis_classifier_infos = [{} for _ in range(self.n_hypothesis)]

        self.hypothesis_probability_history = []
        self.hypothesis_probability_history.append(self.hypothesis_probability.copy())

        self.flash_history = []
        self.signal_history = []

        self.hypothesis_labels = [[] for _ in range(self.n_hypothesis)]

        self.even_flash_patterns = tools.compute_even_flash_patterns(self.n_hypothesis)

    def is_inconsistent(self):
        # this concept deos not realy exist / is impossible to quantify, on continuous cases
        return False

    def is_solved(self):
        return np.max(self.hypothesis_probability) > self.proba_decision_threshold

    def get_solution_index(self):
        if self.is_solved():
            return np.argmax(self.hypothesis_probability)
        else:
            raise Exception('Not Solved yet')

    def enough_labels_per_hypothesis(self):
        for hyp_label in self.hypothesis_labels:
            if not classifier_tools.is_y_valid(hyp_label, N_CLASS_REQUIREMENT, MIN_SAMPLE_PER_CLASS_REQUIREMENT):
                return False
        return True

    def propagate_labels_from_hypothesis(self, hypothesis_index):
        # propagate labels first
        labels_to_propagate = self.hypothesis_labels[hypothesis_index].copy()
        self.hypothesis_labels = [labels_to_propagate.copy() for _ in range(self.n_hypothesis)]

        # we  reset the probabilities for the uncertainty planning
        self.hypothesis_probability = [0.5 for _ in range(self.n_hypothesis)]  # this is not really a probability distribution, see thesis eq 4.10
        # and log it
        self.hypothesis_probability_history.append(self.hypothesis_probability.copy())

        # we recompute the classifiers, for the uncertainty planning
        hypothesis_classifier_infos = []
        for hyp_label in self.hypothesis_labels:
            _, hyp_classifier_info = classifier_tools.compute_loglikelihood(self.signal_history, hyp_label, proba_label_valid=self.proba_assigned_to_label_valid, use_leave_one_out=False)
            hypothesis_classifier_infos.append(hyp_classifier_info)
        self.hypothesis_classifier_infos = hypothesis_classifier_infos

        # we decide that once the first target has been found and we propagate labels, we will stop using leave one out method for computing loglikelihood as all models are now very similar and it would take more and more computational ressource to keep using a leave one out past that point.
        self.use_leave_one_out = False


    def update(self, flash_pattern, feedback_signal):
        # log data
        self.flash_history.append(flash_pattern)
        self.signal_history.append(feedback_signal)

        # update interpretations according to each hypothesis
        for i_hyp in range(self.n_hypothesis):
            self.hypothesis_labels[i_hyp].append(flash_pattern[i_hyp])

        if self.enough_labels_per_hypothesis():

            # compute likelihoods for each hypothesis
            hypothesis_loglikelihoods = []
            hypothesis_classifier_infos = []
            for hyp_label in self.hypothesis_labels:
                hyp_loglikelihood, hyp_classifier_info = classifier_tools.compute_loglikelihood(self.signal_history, hyp_label, proba_label_valid=self.proba_assigned_to_label_valid, use_leave_one_out=self.use_leave_one_out)

                hypothesis_loglikelihoods.append(hyp_loglikelihood)
                hypothesis_classifier_infos.append(hyp_classifier_info)

            # saving classifiers for later reuse in uncertainty calculation
            self.hypothesis_classifier_infos = hypothesis_classifier_infos

            # compute the minimum pairwise normalize likelihood (see eq 4.10 of thesis)
            for i_hyp in range(self.n_hypothesis):
                min_pairwise_norm_likelihood = np.inf
                for j_hyp in range(self.n_hypothesis):
                    # do not compare with itself which would always lead to 0.5
                    if j_hyp != i_hyp:
                        i_loglik = hypothesis_loglikelihoods[i_hyp]
                        j_loglik = hypothesis_loglikelihoods[j_hyp]
                        pairwise_norm_likelihood = classifier_tools.normalize_log_array([i_loglik, j_loglik])[0]

                        if pairwise_norm_likelihood < min_pairwise_norm_likelihood:
                            min_pairwise_norm_likelihood = pairwise_norm_likelihood

                self.hypothesis_probability[i_hyp] = min_pairwise_norm_likelihood

            # fall back method normalizing the all array, less robust and harder to scale
            # self.hypothesis_probability = classifier_tools.normalize_log_array(hypothesis_loglikelihoods).tolist()

        # logging the updated hypothesis_validity
        self.hypothesis_probability_history.append(self.hypothesis_probability.copy())


    def get_next_flash_pattern(self, planning_method='even_uncertainty'):
        if planning_method == 'random':
            return [random.choice([True, False]) for _ in range(self.n_hypothesis)]
        elif planning_method == 'even_random':
            return random.choice(self.even_flash_patterns)
        elif planning_method == 'even_enough_label':
            return random.choice(self.compute_patterns_for_enough_label())
        elif planning_method == 'even_uncertainty':
            # if not enough labesl, the classifiers are not trained so we cannot compute any uncertainty
            if self.enough_labels_per_hypothesis():
                return random.choice(self.compute_uncertain_patterns(self.even_flash_patterns))
            else:
                # we thus select the flashing patterns that will help get enough labels per class for each hypothesis
                return random.choice(self.compute_patterns_for_enough_label(self.even_flash_patterns))

        elif planning_method == 'web_ui':
            # additonal tweaks for making the user experience better

            # make sure the last flash pattern is not used
            # this is only for UI needs so the user can see a different pattern each time to know we have moved from one step and they should send a new feedback signal
            flash_patterns = self.even_flash_patterns
            if len(self.flash_history) > 0:
                last_flash_pattern_shown = self.flash_history[-1]
                flash_patterns = tools.remove_patterns_from_list(flash_patterns, [last_flash_pattern_shown])

            # if not enough labesl, the classifiers are not trained so we cannot compute any uncertainty
            if self.enough_labels_per_hypothesis():

                # get most uncertain patterns
                uncertain_patterns = self.compute_uncertain_patterns(flash_patterns)

                # select the patterns for which the diversify in the labels increase the most across all hypothesis
                selected_patterns = tools.select_high_entropy_patterns(self.n_hypothesis, self.hypothesis_labels, uncertain_patterns)

                # randomly pick from the remaining best flashing patterns
                return random.choice(selected_patterns)
            else:
                # we thus select the flashing patterns that will help get enough labels per class for each hypothesis
                return random.choice(self.compute_patterns_for_enough_label(flash_patterns))

        else:
            raise Exception('Planning method "{}" not defined'.format(method))


    def compute_patterns_for_enough_label(self, flash_patterns):
        """
        We try to see if each pattern helps getting closer to having enough labelled data in all our hypothesis labels to be able to start updating probabilities.

        That is if we see this pattern how much does it helps having N_CLASS_REQUIREMENT and MIN_SAMPLE_PER_CLASS_REQUIREMENT for each hypothesis

        To do so we use a simple scoring system, for each pattern for each hypothesis

        A given patterns scores:
            - 0 points for an hypothesis that is already correctly labelled or if the patterns won't improve the labelling
            - 1 point if improve the labelling for an hypothesis
        """

        pattern_scores = []
        for even_flash_pattern in flash_patterns:

            score = 0
            for i_hyp in range(self.n_hypothesis):
                labels = self.hypothesis_labels[i_hyp]

                # if already ok do nothing (score+=0) else we check what is gained from adding the digit
                if not classifier_tools.is_y_valid(labels, N_CLASS_REQUIREMENT, MIN_SAMPLE_PER_CLASS_REQUIREMENT):

                    future_added_label = even_flash_pattern[i_hyp]
                    if future_added_label in labels:
                        if labels.count(future_added_label) < MIN_SAMPLE_PER_CLASS_REQUIREMENT:
                            # labels in dataset already but not enough samples so it would improve the labelling
                            score += 1
                    else:
                        # not in labels so it would improve the labelling
                        score += 1

            pattern_scores.append(score)

        # we have a score for every patterns now
        # we return an array of the one with best scores
        max_scores = np.max(pattern_scores)
        max_scores_indexes = tools.get_indexes_for_value(pattern_scores, max_scores)
        best_flash_patterns = tools.get_values_at_indexes(flash_patterns, max_scores_indexes)

        return best_flash_patterns


    def compute_uncertain_patterns(self, flash_patterns):
        """
        We use the prediction method as explained in my thesis.

        For each potential patterns and for each hypothesis, we sample a few signals from history and predict their labels given the current classifier for a given hypothesis. We then compare this with the expected label from the task, that is the one the user will give for the same given hypothesis and as a response to the displayed pattern, that is the True of False of the patterns. We compute the entropy of these and compare what each hypothesis expects. The more the hypothesis disagree (weighted by their current likelihood), the more we are expected to learn things by showing the pattern, hence the better.
        """

        # we choose the class order from the first classifier, this is arbitrary
        master_ordered_classes = self.hypothesis_classifier_infos[0]['ordered_classes']

        current_n_samples = len(self.signal_history)
        if current_n_samples <= MAX_N_SAMPLES_TO_COMPUTE_UNCERTAINTY:
            # use all
            samples_indexes = list(range(current_n_samples))
        else:
            # use only MAX_N_SAMPLES_TO_COMPUTE_UNCERTAINTY of them, selected randomly
            samples_indexes = list(range(current_n_samples))
            random.shuffle(samples_indexes)
            samples_indexes = samples_indexes[0:MAX_N_SAMPLES_TO_COMPUTE_UNCERTAINTY]


        pattern_scores = []
        for even_flash_pattern in flash_patterns:

            log_proba_expected_from_pattern = classifier_tools.label_log_proba(even_flash_pattern, master_ordered_classes, self.proba_assigned_to_label_valid)

            # for each sample already collected
            sample_uncertainties = []
            for i_sample in samples_indexes:

                # check the "agreement probabilty", the probability that the predicted signal label and the expected label, see p130 of my thesis
                log_predictions_from_signal = []
                for i_hyp in range(self.n_hypothesis):
                    log_pred_from_signal = self.hypothesis_classifier_infos[i_hyp]['log_y_pred'][i_sample]

                    # we reposition the columns wrt master_ordered_classes just in case we run into this issue: https://github.com/scikit-learn/scikit-learn/issues/13211#issuecomment-511392497
                    # this was fixed by using a calibrator so it should never happen
                    current_classes = self.hypothesis_classifier_infos[i_hyp]['ordered_classes']
                    ordered_log_pred_from_signal = classifier_tools.reorder_column_according_to_target_classes_ordering(log_pred_from_signal, current_classes, master_ordered_classes)

                    log_predictions_from_signal.append(ordered_log_pred_from_signal)

                # likelihood that the classifier output matches with the labels:
                # prod_i sum_y P(y_true_{i}=y)P(y_pred_{i}=y)
                # but done all in log to avoid numerical issues
                joint_loglikelihood_per_hypothesis = classifier_tools.sum_log_array(log_predictions_from_signal + log_proba_expected_from_pattern)

                sample_uncertainty = classifier_tools.weighted_variance(np.exp(joint_loglikelihood_per_hypothesis), self.hypothesis_probability)
                sample_uncertainties.append(sample_uncertainty)

            pattern_scores.append(np.sum(sample_uncertainties))

        # we have an uncertain score for every patterns now
        # we return an array of the most uncertain one
        max_scores = np.max(pattern_scores)
        max_scores_indexes = tools.get_indexes_for_value(pattern_scores, max_scores)
        best_flash_patterns = tools.get_values_at_indexes(flash_patterns, max_scores_indexes)

        # import IPython; IPython.embed()

        return best_flash_patterns


    def get_logs(self):

        logs = {}

        logs['learner_type'] = 'continuous'

        logs['n_hypothesis'] = self.n_hypothesis

        logs['use_leave_one_out'] = self.use_leave_one_out
        logs['proba_decision_threshold'] = self.proba_decision_threshold
        logs['proba_assigned_to_label_valid'] = self.proba_assigned_to_label_valid

        logs['hypothesis_probability_history'] = self.hypothesis_probability_history

        logs['flash_history'] = self.flash_history
        logs['signal_history'] = self.signal_history

        logs['hypothesis_labels'] = self.hypothesis_labels

        logs['is_inconsistent'] = bool(self.is_inconsistent())  # bool(for json serialisation)
        logs['is_solved'] = bool(self.is_solved())
        if self.is_solved():
            logs['solution_index'] = int(self.get_solution_index())

        return logs
