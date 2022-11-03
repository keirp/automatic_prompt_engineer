from automatic_prompt_engineer import config


def test_nested_update():
    user_config = {
        'generation': {
            'model': {
                'gpt_config': {
                    'temperature': 1.0
                }
            }
        }
    }

    updated_config = config.update_config(user_config)

    import os
    import yaml
    with open(os.path.join(os.path.dirname(__file__), '../automatic_prompt_engineer/configs/default.yaml')) as f:
        manually_updated_config = yaml.safe_load(f)
    manually_updated_config['generation']['model']['gpt_config']['temperature'] = 1.0

    # Assert that the manually updated config is the same as the updated config (recursively)
    def assert_equal(d1, d2):
        for k, v in d1.items():
            if isinstance(v, dict):
                assert_equal(v, d2[k])
            else:
                assert v == d2[k]

    assert_equal(updated_config, manually_updated_config)
