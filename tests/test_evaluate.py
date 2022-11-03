from automatic_prompt_engineer import config, llm, template
from automatic_prompt_engineer.evaluation.likelihood import likelihood_evaluator, get_query


def test_likelihood():
    """need prompts, eval_template, eval_data, config"""
    prompts = ["Solve the math equation", "Solve the math problem"]
    eval_template = "Task: [PROMPT]\n\n Q: [INPUT] A: [OUTPUT]"
    demos_template = "Q: [INPUT]\nA: [OUTPUT]"
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    eval_data = ["1 + 1", "2 + 2"], ["2", "4"]
    user_config = {
        'evaluation': {
            'num_samples': 2,
            'num_few_shot': 1,
            'model': {
                'gpt_config': {
                    'model': 'text-ada-001'
                }
            }
        }
    }
    conf = config.update_config(user_config)
    res = likelihood_evaluator(
        prompts, eval_template, eval_data, demos_template, eval_data, conf['evaluation'])
    prompts, scores = res.sorted()
    assert len(scores) == len(prompts)


def test_likelihood_indices():
    """need prompts, eval_template, eval_data, config"""
    prompts = ["Solve the math equation", "Solve the math problem"]
    eval_template = "Task: [PROMPT]\n\n Q: [INPUT] A: [OUTPUT]"
    demos_template = "Q: [INPUT]\nA: [OUTPUT]"
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    eval_data = ["1 + 1", "2 + 2"], ["2", "4"]
    user_config = {
        'evaluation': {
            'num_samples': 2,
            'model': {
                'batch_size': 1,
                'gpt_config': {
                    'model': 'text-ada-001'
                }
            }
        }
    }
    conf = config.update_config(user_config)
    queries = []
    output_indices = []
    for prompt in prompts:
        for data in zip(*eval_data):
            input_, output_ = data
            query, output_idx = get_query(
                prompt, eval_template, input_, output_, eval_data, demos_template)
            queries.append(query)
            output_indices.append(output_idx)

    for i, (query, output_idx) in enumerate(zip(queries, output_indices)):
        assert query[output_idx[0]:output_idx[1]] in eval_data[1]

    # Instantiate the LLM
    model = llm.model_from_config(conf['evaluation']['model'])

    _, tokens = model.log_probs(queries, output_indices)
    reconstructed_tokens = [''.join(tokens[i]).strip()
                            for i in range(len(tokens))]
    for i, (_, output) in enumerate(zip(*eval_data)):
        assert output == reconstructed_tokens[i]
