import random
from tqdm import tqdm
from abc import ABC, abstractmethod

import numpy as np

from automatic_prompt_engineer import evaluate


def bandits_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    base_eval_method = evaluate.get_eval_method(config['base_eval_method'])
    bandit_algo = get_bandit_algo(
        config['bandit_method'], len(prompts), config)
    rounds = config['rounds']
    if config['num_prompts_per_round'] < 1:
        num_prompts_per_round = int(
            len(prompts) * config['num_prompts_per_round'])
    else:
        num_prompts_per_round = config['num_prompts_per_round']
    num_prompts_per_round = min(num_prompts_per_round, len(prompts))
    for _ in tqdm(range(rounds), desc='Evaluating prompts'):
        # Sample the prompts
        sampled_prompts_idx = bandit_algo.choose(num_prompts_per_round)
        sampled_prompts = [prompts[i] for i in sampled_prompts_idx]
        # Evaluate the sampled prompts
        sampled_eval_results = base_eval_method(
            sampled_prompts, eval_template, eval_data, demos_template, few_shot_data, config['base_eval_config'])
        _, scores = sampled_eval_results.in_place(method='mean')
        # Update the bandit algorithm
        bandit_algo.update(sampled_prompts_idx, scores)

    return BanditsEvaluationResult(prompts, bandit_algo.get_scores(), bandit_algo.get_infos())


def get_bandit_algo(bandit_method, num_prompts, config):
    """
    Returns the bandit method object.
    Parameters:
        bandit_method: The bandit method to use. ('epsilon-greedy')
    Returns:
        A bandit method object.
    """
    if bandit_method == 'ucb':
        return UCBBanditAlgo(num_prompts, config['base_eval_config']['num_samples'], config['bandit_config']['c'])
    else:
        raise ValueError('Invalid bandit method.')


class BanditsEvaluationResult(evaluate.EvaluationResult):

    def __init__(self, prompts, scores, infos):
        self.prompts = prompts
        self.scores = scores
        self.infos = infos

    def sorted(self, method='default'):
        """Sort the prompts and scores. There is no choice of method for now."""
        idx = np.argsort(self.scores)
        prompts, scores = [self.prompts[i]
                           for i in idx], [self.scores[i] for i in idx]
        # Reverse
        prompts, scores = prompts[::-1], scores[::-1]
        return prompts, scores

    def in_place(self, method='default'):
        """Return the prompts and scores in place. There is no choice of method for now."""
        return self.prompts, self.scores

    def sorted_infos(self):
        """Sort the infos."""
        idx = np.argsort(self.scores)
        infos = [self.infos[i] for i in idx]
        # Reverse
        infos = infos[::-1]
        return infos

    def __str__(self):
        s = ''
        prompts, scores = self.sorted()
        s += 'score: prompt\n'
        s += '----------------\n'
        for prompt, score in list(zip(prompts, scores))[:10]:
            s += f'{score:.2f}: {prompt}\n'
        return s


class BatchBanditAlgo(ABC):

    @ abstractmethod
    def choose(self, n):
        """Choose n prompts from the scores.
        Parameters:
            n: The number of prompts to choose.
        Returns:
            A list of indices of the chosen prompts.
        """
        pass

    @ abstractmethod
    def update(self, chosen, scores):
        """Update the scores for the chosen prompts.
        Parameters:
            chosen: A list of indices of the chosen prompts.
            scores: A list of scores for each chosen prompt in the form of a list.
        """
        pass

    @ abstractmethod
    def reset(self):
        """Reset the algorithm."""
        pass

    @ abstractmethod
    def get_scores(self):
        """Get the scores for all prompts.
        Returns:
            A list of scores.
        """
        pass

    @ abstractmethod
    def get_infos(self):
        """Get the infos for all prompts.
        Returns:
            A list of infos.
        """
        pass


class CountAverageBanditAlgo(BatchBanditAlgo):

    def __init__(self, num_prompts, num_samples):
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.reset()

    def update(self, chosen, scores):
        for i, score in zip(chosen, scores):
            self.counts[i] += self.num_samples
            self.scores[i] += score * self.num_samples

    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def get_scores(self):
        # Some counts may be 0, so we need to avoid division by 0.
        return np.divide(self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)


class UCBBanditAlgo(CountAverageBanditAlgo):

    def __init__(self, num_prompts, num_samples, c):
        super().__init__(num_prompts, num_samples)
        self.c = c

    def choose(self, n):
        if np.sum(self.counts) == 0:
            # If all counts are 0, choose randomly.
            return random.sample(range(self.num_prompts), n)
        scores = self.get_scores()
        counts = self.counts + 1e-3
        ucb_scores = scores + self.c * np.sqrt(np.log(np.sum(counts)) / counts)
        # Choose the prompts with the highest UCB scores
        return np.argsort(ucb_scores)[::-1][:n]

    def get_infos(self):
        return self.counts
