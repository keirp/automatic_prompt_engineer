import pandas as pd

induce_data_path = 'experiments/data/TruthfulQA/TruthfulQA_train.csv'
eval_data_path = 'experiments/data/TruthfulQA/TruthfulQA_test.csv'


def load_data(type):
    base_path = induce_data_path if type == 'induce' else eval_data_path
    path = base_path
    with open(path, 'r') as f:
        data = pd.read_csv(f)
        # print the column names
        input_, output_ = data['Question'], data['Best Answer']
    # Convert to list
    input_ = input_.tolist()
    output_ = output_.tolist()
    return input_, output_
