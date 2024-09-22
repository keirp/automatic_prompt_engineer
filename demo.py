import openai
import numpy as np
import pandas as pd
import gradio as gr

from experiments.data.instruction_induction.load_data import load_data
from automatic_prompt_engineer.ape import get_simple_prompt_gen_template
from automatic_prompt_engineer import ape, evaluate, config, template, llm

model_types = ['text-ada-001', 'text-babbage-001', 'text-curie-001', 'gpt-3.5-turbo']
mode_types = ['forward', 'insert']
eval_types = ['likelihood', 'bandits']
task_types = ['antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
              'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
              'active_to_passive', 'singular_to_plural', 'rhymes',
              'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
              'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
              'translation_en-fr', 'word_in_context']


def load_task(task):
    inputs, outputs = load_data('induce', task)
    train_data = '\n'.join([f'{inp} >>> {out[0] if len(out) == 1 else out}' for inp, out in zip(inputs, outputs)])

    inputs, outputs = load_data('execute', task)
    test_data = '\n'.join([f'{inp} >>> {out[0] if len(out) == 1 else out}' for inp, out in zip(inputs, outputs)])

    return train_data, test_data


# Problem dataset: [common_concept, larger_animal,taxonomy_animal,rhymes,sentiment,orthography_starts_with]
def parse_data(dataset):
    """Parses the data from the text of a csv into two lists of strings.
    The CSV has no header and the first column is the input and the second column is the output."""
    dataset = dataset.split('\n')
    dataset = [line.split(' >>> ') for line in dataset]
    dataset = [inp for inp, _ in dataset], [out for inp, out in dataset]
    return dataset


def prompts_to_df(prompts, scores):
    """Converts a list of prompts into a dataframe."""
    df = pd.DataFrame()
    df['Prompt'] = prompts
    df['log(p)'] = scores
    df['log(p)'] = df['log(p)'].apply(lambda x: round(x, 3))  # Round the scores to 3 decimal places
    df = df.head(15)  # Only show the top 15 prompts
    return df


def prompt_overview():
    gen_prompt = ''
    eval_prompt = ''

    return gen_prompt, eval_prompt


def run_ape(prompt_gen_data, eval_data,
            eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
            prompt_gen_template=None,
            demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
            eval_model='gpt-3.5-turbo',
            prompt_gen_model='gpt-3.5-turbo',
            prompt_gen_mode='forward',
            num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500,  # basic
            num_subsamples=None, num_demos=None,  # advanced
            num_samples=None, num_few_shot=None  # advanced
            ):
    """
    Function that wraps the find_prompts function to make it easier to use.
    Design goals: include default values for most parameters, and automatically
    fill out the config dict for the user in a way that fits almost all use cases.

    The main shortcuts this function takes are:
    - Uses the same dataset for prompt generation, evaluation, and few shot demos
    - Uses UCB algorithm for evaluation
    - Fixes the number of prompts per round to num_prompts // 3  (so the first three rounds will
        sample every prompt once)
    - Fixes the number of samples per prompt per round to 5
    Parameters:
        prompt_gen_data: The dataset to use for prompt generation.
        eval_data: The dataset to use for evaluation.
        eval_template: The template for the evaluation queries.
        prompt_gen_template: The template to use for prompt generation.
        demos_template: The template for the demos.
        eval_model: The model to use for evaluation.
        prompt_gen_model: The model to use for prompt generation.
        prompt_gen_mode: The mode to use for prompt generation.
        num_prompts: The number of prompts to generate during the search.
        eval_rounds: The number of evaluation rounds to run.
        prompt_gen_batch_size: The batch size to use for prompt generation.
        eval_batch_size: The batch size to use for evaluation.
        num_subsamples: The number of different demos to generate the instruction.
        num_demos: The number of demos to generate the instruction.
        num_samples: The number of samples at each round for evaluation.
        num_few_shot: The number of few shot examples to use for evaluation.
    Returns:
        An evaluation result and a function to evaluate the prompts with new inputs.
    """
    prompt_gen_data = parse_data(prompt_gen_data)
    eval_data = parse_data(eval_data)
    prompt_gen_template = get_simple_prompt_gen_template(prompt_gen_template, prompt_gen_mode)

    if demos_template is None:
        demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    conf = config.simple_config(
        eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size)

    if num_subsamples is not None:
        conf['generation']['num_subsamples'] = num_subsamples
        conf['generation']['num_prompts_per_subsample'] = num_prompts // num_subsamples

    if num_demos is not None:
        conf['generation']['num_demos'] = num_demos

    if num_samples is not None:
        if conf['evaluation']['method'] == 'bandits':
            conf['evaluation']['base_eval_config']['num_samples'] = num_samples
        else:
            conf['evaluation']['num_samples'] = num_samples

    if num_few_shot is not None:
        if conf['evaluation']['method'] == 'bandits':
            conf['evaluation']['base_eval_config']['num_few_shot'] = num_few_shot
        else:
            conf['evaluation']['num_few_shot'] = num_few_shot

    res, demo_fn = ape.find_prompts(eval_template, demos_template, prompt_gen_data, eval_data, conf,
                                    prompt_gen_template=prompt_gen_template)

    prompts, scores = res.sorted()

    df = prompts_to_df(prompts, scores)
    generation_query = ape.get_generation_query(eval_template, demos_template, conf, prompt_gen_data,
                                                prompt_gen_template)[0]
    evaluation_query = ape.get_evaluation_query(eval_template, demos_template, conf, eval_data, prompt_gen_data)[0]

    return df, generation_query, evaluation_query, prompts[0], prompts[0], scores[0]


def basic_ape(prompt_gen_data, eval_data,
              eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
              eval_model='gpt-3.5-turbo',
              prompt_gen_model='gpt-3.5-turbo',
              prompt_gen_mode='forward',
              num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500):
    return run_ape(prompt_gen_data, eval_data, eval_template,
                   eval_model=eval_model, prompt_gen_model=prompt_gen_model, prompt_gen_mode=prompt_gen_mode,
                   num_prompts=num_prompts, eval_rounds=eval_rounds, prompt_gen_batch_size=prompt_gen_batch_size,
                   eval_batch_size=eval_batch_size)


def advance_ape(prompt_gen_data, eval_data,
                eval_template, prompt_gen_template, demos_template,
                eval_model='gpt-3.5-turbo',
                prompt_gen_model='gpt-3.5-turbo',
                prompt_gen_mode='forward',
                num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500,  # basic
                num_subsamples=None, num_demos=None,  # advanced
                num_samples=None, num_few_shot=None  # advanced
                ):
    if prompt_gen_mode == 'forward':
        if prompt_gen_template[-5:] != '[APE]':
            raise ValueError('The prompt_gen_template must end with [APE] for forward mode.')

    return run_ape(prompt_gen_data, eval_data, eval_template, prompt_gen_template, demos_template,
                   eval_model=eval_model, prompt_gen_model=prompt_gen_model, prompt_gen_mode=prompt_gen_mode,
                   num_prompts=num_prompts, eval_rounds=eval_rounds, prompt_gen_batch_size=prompt_gen_batch_size,
                   eval_batch_size=eval_batch_size, num_subsamples=num_subsamples, num_demos=num_demos,
                   num_samples=num_samples, num_few_shot=num_few_shot)


def estimate_cost(prompt_gen_data, eval_data,
                  eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
                  prompt_gen_template=None,
                  demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
                  eval_model='gpt-3.5-turbo',
                  prompt_gen_model='gpt-3.5-turbo',
                  prompt_gen_mode='forward',
                  num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500,  # basic
                  num_subsamples=None, num_demos=None,  # advanced
                  num_samples=None, num_few_shot=None  # advanced
                  ):
    """
    Function that wraps the find_prompts function to make it easier to use.
    Design goals: include default values for most parameters, and automatically
    fill out the config dict for the user in a way that fits almost all use cases.

    The main shortcuts this function takes are:
    - Uses the same dataset for prompt generation, evaluation, and few shot demos
    - Uses UCB algorithm for evaluation
    - Fixes the number of prompts per round to num_prompts // 3  (so the first three rounds will
        sample every prompt once)
    - Fixes the number of samples per prompt per round to 5
    Parameters:
        prompt_gen_data: The dataset to use for prompt generation.
        eval_data: The dataset to use for evaluation.
        eval_template: The template for the evaluation queries.
        prompt_gen_template: The template to use for prompt generation.
        demos_template: The template for the demos.
        eval_model: The model to use for evaluation.
        prompt_gen_model: The model to use for prompt generation.
        prompt_gen_mode: The mode to use for prompt generation.
        num_prompts: The number of prompts to generate during the search.
        eval_rounds: The number of evaluation rounds to run.
        prompt_gen_batch_size: The batch size to use for prompt generation.
        eval_batch_size: The batch size to use for evaluation.
        num_subsamples: The number of different demos to generate the instruction.
        num_demos: The number of demos to generate the instruction.
        num_samples: The number of samples at each round for evaluation.
        num_few_shot: The number of few shot examples to use for evaluation.
    Returns:
        An evaluation result and a function to evaluate the prompts with new inputs.
    """
    prompt_gen_data = parse_data(prompt_gen_data)
    eval_data = parse_data(eval_data)
    prompt_gen_template = get_simple_prompt_gen_template(prompt_gen_template, prompt_gen_mode)
    if demos_template is None:
        demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    conf = config.simple_config(
        eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size)
    if num_subsamples is not None:
        conf['generation']['num_subsamples'] = num_subsamples
        conf['generation']['num_prompts_per_subsample'] = num_prompts // num_subsamples

    if num_demos is not None:
        conf['generation']['num_demos'] = num_demos

    if num_samples is not None:
        if conf['evaluation']['method'] == 'bandits':
            conf['evaluation']['base_eval_config']['num_samples'] = num_samples
        else:
            conf['evaluation']['num_samples'] = num_samples

    if num_few_shot is not None:
        if conf['evaluation']['method'] == 'bandits':
            conf['evaluation']['base_eval_config']['num_few_shot'] = num_few_shot
        else:
            conf['evaluation']['num_few_shot'] = num_few_shot

    cost = ape.estimate_cost(eval_template, demos_template, prompt_gen_data, eval_data, conf,
                             prompt_gen_template=prompt_gen_template)
    generation_query = ape.get_generation_query(eval_template, demos_template, conf, prompt_gen_data,
                                                prompt_gen_template)[0]
    evaluation_query = ape.get_evaluation_query(eval_template, demos_template, conf, eval_data, prompt_gen_data)[0]
    return cost, generation_query, evaluation_query


def basic_estimate_cost(prompt_gen_data,
                        eval_data,
                        eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
                        eval_model='gpt-3.5-turbo',
                        prompt_gen_model='gpt-3.5-turbo',
                        prompt_gen_mode='forward',
                        num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500):
    return estimate_cost(prompt_gen_data, eval_data, eval_template,
                         eval_model=eval_model, prompt_gen_model=prompt_gen_model, prompt_gen_mode=prompt_gen_mode,
                         num_prompts=num_prompts, eval_rounds=eval_rounds, prompt_gen_batch_size=prompt_gen_batch_size,
                         eval_batch_size=eval_batch_size)


def advance_estimate_cost(prompt_gen_data, eval_data,
                          eval_template, prompt_gen_template, demos_template,
                          eval_model='gpt-3.5-turbo',
                          prompt_gen_model='gpt-3.5-turbo',
                          num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500,  # basic
                          num_subsamples=None, num_demos=None,  # advanced
                          num_samples=None, num_few_shot=None  # advanced
                          ):
    if prompt_gen_template[-5:] == '[APE]':
        prompt_gen_mode = 'forward'
    else:
        prompt_gen_mode = 'insert'

    return estimate_cost(prompt_gen_data, eval_data, eval_template, prompt_gen_template, demos_template,
                         eval_model=eval_model, prompt_gen_model=prompt_gen_model, prompt_gen_mode=prompt_gen_mode,
                         num_prompts=num_prompts, eval_rounds=eval_rounds, prompt_gen_batch_size=prompt_gen_batch_size,
                         eval_batch_size=eval_batch_size, num_subsamples=num_subsamples, num_demos=num_demos,
                         num_samples=num_samples, num_few_shot=num_few_shot)


def compute_score(prompt,
                  eval_data,
                  eval_template,
                  demos_template,
                  eval_model='gpt-3.5-turbo',
                  num_few_shot=None  # advanced
                  ):
    eval_data = parse_data(eval_data)
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    conf = config.update_config({}, 'configs/default.yaml')
    conf['evaluation']['model']['gpt_config']['model'] = eval_model
    conf['evaluation']['num_samples'] = min(len(eval_data[0]), 50)
    conf['evaluation']['num_few_shot'] = num_few_shot
    res = evaluate.evalute_prompts([prompt], eval_template, eval_data, demos_template, eval_data,
                                   conf['evaluation']['method'], conf['evaluation'])
    return round(res.sorted()[1][0], 3)


def run_prompt(prompt, inputs,
               eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
               eval_model='gpt-3.5-turbo',
               prompt_gen_model='gpt-3.5-turbo',
               prompt_gen_mode='forward',
               num_prompts=50, eval_rounds=10, prompt_gen_batch_size=200, eval_batch_size=500):
    conf = config.simple_config(
        eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size)
    eval_template = template.EvalTemplate(eval_template)
    model = llm.model_from_config(conf['evaluation']['base_eval_config']['model'])

    if inputs == '':
        queries = [prompt]
    else:
        if not isinstance(inputs, list):
            inputs = [inputs]
        queries = []
        for input_ in inputs:
            query = eval_template.fill(prompt=prompt, input=input_)
            queries.append(query)

    outputs = model.generate_text(queries, n=1)
    return outputs[0].strip()


def get_demo():
    assert openai.api_key is not None, 'Please set your OpenAI API key first.'
    assert openai.api_key != '', 'Please set your OpenAI API key first.'

    with gr.Blocks(title="Automatic Prompt Engineer", css=None, ) as demo:
        gr.Markdown("# Automatic Prompt Engineer")
        gr.Markdown("""This WebUI demonstrates how to use Automatic Prompt Engineer [APE](arxiv link) to optimize 
            prompts for text generation. In its simplest form, APE takes as input a dataset (a list of inputs and a 
            list of  outputs), a prompt template, and optimizes this prompt template so that it generates the outputs 
            given the inputs. APE accomplishes this in two steps. First, it uses a language model to generate a set of 
            candidate prompts. Then, it uses a prompt evaluation function to evaluate the quality of each candidate 
            prompt. Finally, it returns the prompt with the highest evaluation score. Let's try it out!""")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## APE")
                with gr.Tab("Basic"):
                    prompt_gen_mode = gr.Dropdown(label="Prompt Generation Mode", choices=mode_types,
                                                  value="forward")
                    with gr.Row():
                        eval_template = gr.Textbox(lines=5,
                                                   value="Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]",
                                                   label="Evaluation Template")
                    with gr.Row():
                        basic_cost = gr.Textbox(lines=1, value="", label="Estimated Cost ($)", disabled=True)
                        basic_cost_button = gr.Button("Estimate Cost")
                        basic_ape_button = gr.Button("APE")

                with gr.Tab("Advanced"):
                    with gr.Row():
                        prompt_gen_template = gr.Textbox(lines=6,
                                                         value="""I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]""",
                                                         label="Prompt Generation Template")
                        demos_template = gr.Textbox(lines=6, value="Input: [INPUT]\nOutput: [OUTPUT]",
                                                    label="Demos Template")

                    with gr.Row():
                        cost = gr.Textbox(lines=1, value="", label="Estimated Cost ($)", disabled=True)
                        cost_button = gr.Button("Estimate Cost")
                        ape_button = gr.Button("APE")

                with gr.Tab("Dataset"):
                    with gr.Row():
                        prompt_gen_data = gr.Textbox(lines=10, value=load_task('antonyms')[0],
                                                     label="Data for prompt generation")
                        eval_data = gr.Textbox(lines=10, value=load_task('antonyms')[1],
                                               label="Data for scoring")
                    with gr.Row():
                        task = gr.Dropdown(label="Task", choices=task_types, value="antonyms")
                        load_task_button = gr.Button("Load Task")

            with gr.Column(scale=2):
                gr.Markdown("## Configuration")
                with gr.Tab("Basic"):
                    with gr.Row():
                        prompt_gen_model = gr.Dropdown(label="Prompt Generation Model", choices=model_types,
                                                       value="gpt-3.5-turbo")

                        eval_model = gr.Dropdown(label="Evaluation Model", choices=model_types,
                                                 value="gpt-3.5-turbo")

                    with gr.Row():
                        num_prompts = gr.Slider(label="Number of Prompts", minimum=1, maximum=250, step=10, value=50)
                        eval_rounds = gr.Slider(label="Rounds of Evaluation", minimum=1, maximum=20, step=1, value=5)

                    with gr.Row():
                        prompt_gen_batch_size = gr.Slider(label="Batch Size for prompt generation",
                                                          minimum=1, maximum=500, step=10, value=200)
                        eval_batch_size = gr.Slider(label="Batch Size for Evaluation",
                                                    minimum=4, maximum=1000, step=1, value=500)

                with gr.Tab("Advanced"):
                    with gr.Column(scale=1):
                        with gr.Row():
                            num_subsamples = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                                       label="Number of subsamples for instruction generation")
                            num_demos = gr.Slider(minimum=1, maximum=25, value=5, step=1,
                                                  label="Number of demos for instruction generation")

                        with gr.Row():
                            num_samples = gr.Slider(minimum=1, maximum=50, value=10, step=1,
                                                    label="Number of evaluation samples at each round")
                            num_few_shot = gr.Slider(minimum=0, maximum=25, value=0, step=1,
                                                     label="Number of few-shot examples used in evaluation")

        gr.Markdown("## Results")
        with gr.Row():
            with gr.Tab("APE Results"):
                output_df = gr.DataFrame(type='pandas', headers=['Prompt', 'Likelihood'], wrap=True, interactive=False)

            with gr.Tab("Prompt Overview"):
                with gr.Row():
                    generation_prompt_sample = gr.Textbox(lines=8, value="",
                                                          label="Instruction Generation Prompts",
                                                          disabled=True)
                    evaluation_prompt_sample = gr.Textbox(lines=8, value="",
                                                          label="Evaluation Prompts",
                                                          disabled=True)

            with gr.Tab("Prompt Deployment"):
                with gr.Row():
                    with gr.Column(scale=1):
                        test_prompt = gr.Textbox(lines=4, value="Please evaluate the following expression.",
                                                 label="Prompt")
                        test_inputs = gr.Textbox(lines=1, value="",
                                                 label="Input (If empty, prompt is executed directly)")
                        answer_button = gr.Button("Submit")
                    with gr.Column(scale=1):
                        test_output = gr.Textbox(lines=9, value="", label="Model Output")

            with gr.Tab("Prompt Score"):
                with gr.Row():
                    with gr.Column(scale=1):
                        score_prompt = gr.Textbox(lines=3, value="Please evaluate the following expression.",
                                                  label="Prompt (Evaluate on scoring dataset using Evaluation Template)")
                        compute_score_button = gr.Button("Compute Score")
                    with gr.Column(scale=1):
                        test_score = gr.Textbox(lines=1, value="", label="Log(p)", disabled=True)

        ##############################
        # Button Callbacks
        ##############################
        load_task_button.click(load_task, inputs=[task], outputs=[prompt_gen_data, eval_data])
        basic_ape_button.click(basic_ape,
                               inputs=[prompt_gen_data, eval_data,  # Data
                                       eval_template,  # Templates
                                       eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds,
                                       prompt_gen_batch_size, eval_batch_size,  # Basic Configuration
                                       ],
                               outputs=[output_df, generation_prompt_sample, evaluation_prompt_sample,
                                        test_prompt, score_prompt, test_score])
        ape_button.click(advance_ape,
                         inputs=[prompt_gen_data, eval_data,  # Data
                                 eval_template, prompt_gen_template, demos_template,  # Templates
                                 eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds,
                                 prompt_gen_batch_size, eval_batch_size,  # Basic
                                 num_subsamples, num_demos, num_samples, num_few_shot  # Advanced
                                 ],
                         outputs=[output_df, generation_prompt_sample, evaluation_prompt_sample,
                                  test_prompt, score_prompt, test_score])

        basic_cost_button.click(basic_estimate_cost,
                                inputs=[prompt_gen_data, eval_data,  # Data
                                        eval_template,  # Templates
                                        eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds,
                                        prompt_gen_batch_size, eval_batch_size,  # Basic Configuration
                                        ],
                                outputs=[basic_cost, generation_prompt_sample, evaluation_prompt_sample])

        cost_button.click(advance_estimate_cost,
                          inputs=[prompt_gen_data, eval_data,  # Data
                                  eval_template, prompt_gen_template, demos_template,  # Templates
                                  eval_model, prompt_gen_model, num_prompts, eval_rounds,
                                  prompt_gen_batch_size, eval_batch_size,  # Basic
                                  num_subsamples, num_demos, num_samples, num_few_shot  # Advanced
                                  ],
                          outputs=[cost, generation_prompt_sample, evaluation_prompt_sample])

        compute_score_button.click(compute_score,
                                   inputs=[score_prompt,
                                           eval_data,
                                           eval_template,
                                           demos_template,
                                           eval_model,
                                           num_few_shot],
                                   outputs=[test_score])

        answer_button.click(run_prompt,
                            inputs=[test_prompt, test_inputs, eval_template, eval_model,
                                    prompt_gen_model, prompt_gen_mode,
                                    num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size],
                            outputs=[test_output])

        return demo


if __name__ == "__main__":
    demo = get_demo()
    demo.launch(show_error=True)
