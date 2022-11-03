import os
import json
import random

induce_data_path = os.path.join(os.path.dirname(__file__), 'raw/induce/')
eval_data_path = os.path.join(os.path.dirname(__file__), 'raw/execute/')

# Get a list of tasks (by looking at the names of the files in the induced directory)
tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]


def load_data(type, task):
    base_path = induce_data_path if type == 'induce' else eval_data_path
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        if task == 'cause_and_effect':
            cause, effect = data['cause'], data['effect']
            # Pick an order randomly
            if random.random() < 0.5:
                input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
            else:
                input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
            output_ = [cause]
        elif task == 'common_concept':
            items = data['items']
            # Make comma separated list of items
            input_ = ', '.join(items[:-1])
            output_ = data['all_common_concepts']
        elif task == 'rhymes':
            input_, output_ = data['input'], data['other_rhymes']
        elif 'translation' in task:
            input_, output_ = data['input'], data['possible_translations']
        else:
            input_, output_ = data['input'], [data['output']]
        inputs.append(input_)
        outputs.append(output_)
    return inputs, outputs
