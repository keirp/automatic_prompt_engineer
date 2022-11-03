import numpy as np

from automatic_prompt_engineer import data, llm, evaluate
from experiments.evaluation.instruction_induction import utility


def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)
    return query


def exec_accuracy_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    queries = []
    answers = []
    for prompt in prompts:
        subsampled_data = data.subsample_data(
            eval_data, config['num_samples'])
        for d in zip(*subsampled_data):
            input_, output_ = d
            demo_data = data.subsample_data(
                few_shot_data, config['num_few_shot'])
            query = get_query(
                prompt, eval_template, input_, output_, demo_data, demos_template)
            queries.append(query)
            answers.append(output_)

    # Instantiate the LLM
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1)

    task = config['task']
    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)

    print(f'Using metric "{metric}" for task "{task}"...')

    if metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em

    scores = []
    for prediction, ans_ in zip(model_outputs, answers):
        score = score_fn(prediction, ans_)
        scores.append(score)

    # Reshape the scores so that it is num_prompts x num_samples
    scores = np.array(scores).reshape(len(prompts), config['num_samples'])

    res = ExecAccuracyEvaluationResult(prompts, scores)
    return res


class ExecAccuracyEvaluationResult(evaluate.EvaluationResult):

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        if method == 'mean':
            return [np.mean(s) for s in self.scores]
        elif method == 'median':
            return [np.median(s) for s in self.scores]
        elif method == 'std':
            return [np.std(s) for s in self.scores]
        elif method == 'max':
            return [np.max(s) for s in self.scores]
        elif method == 'min':
            return [np.min(s) for s in self.scores]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.scores]
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        return self.prompts, scores
