from automatic_prompt_engineer import ape, data
from experiments.data.TruthfulQA.load_data import load_data
from experiments.evaluation.TruthfulQA.fine_tuned_gpt_eval import fine_tuned_gpt_evaluator


def indented_print(s, indent=2):
    s = s.replace('\n', '\n' + ' ' * indent)
    print(' ' * indent + s)


def utility_combination(truthful, informative):
    if truthful == 1 and informative == 1:
        return 1
    elif truthful == 1 and informative == 0:
        return 0.5
    return 0


def run():
    induce_data, test_data = load_data('induce'), load_data('eval')
    prompt_gen_size = 20

    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)

    eval_template = "Instructions: [PROMPT].\n\nQ: [INPUT]\nA: [OUTPUT]"
    prompt_gen_template = "You are given the following instructions: [APE]. Now please answer the following questions.\n\n[full_DEMO]"
    demos_template = "Q: [INPUT]\nA: [OUTPUT]"

    base_config = '../experiments/configs/truthful_qa.yaml'
    conf = {
        'generation': {
            'num_subsamples': 9,
            'num_demos': 5,
            'num_prompts_per_subsample': 10,
            'model': {
                'name': 'GPT_insert',
                'batch_size': 1,
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': fine_tuned_gpt_evaluator,
            'num_samples': 10,
            'model': {
                'gpt_config': {
                    # 'model': 'text-curie-001'
                }
            }
        }
    }

    res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=eval_data,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=None,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)

    print('Finished finding prompts.')

    combinations = ['truthful', 'informative', 'both', utility_combination]

    for comb in combinations:

        prompts, scores = res.sorted(combination=comb)
        # get comb name (use function name if function)
        comb_name = comb.__name__ if callable(comb) else comb
        print(f'Prompts (ranked by {comb_name}):')
        for prompt, score in list(zip(prompts, scores))[:10]:
            indented_print(f'{score}: {prompt}')

        # Save the prompts and scores
        with open(f'experiments/results/truthful_qa/{comb_name}.txt', 'w') as f:
            for prompt, score in list(zip(prompts, scores)):
                f.write(f'Score: {score:.2f}\nPrompt:{prompt}\n')


if __name__ == '__main__':
    run()
