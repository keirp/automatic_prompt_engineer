from automatic_prompt_engineer import llm, config


def test_log_prob():
    """Tests the log prob function."""
    user_config = {
        'evaluation': {
            'model': {
                'gpt_config': {
                    'model': 'text-ada-001'
                }
            }
        }
    }
    conf = config.update_config(user_config)
    model = llm.model_from_config(conf['evaluation']['model'])
    test_inputs = ["Solve the math equation", "Solve the math problem"]
    # Get the indices of the word "equation" and "problem"

    log_probs, tokens = model.log_probs(test_inputs, [(14, 23), (14, 22)])
    reconstructed_tokens = ''.join(tokens[0])
    assert reconstructed_tokens == test_inputs[0][14:23]
    assert len(log_probs) == 2


def test_strange_character():
    user_config = {
        'evaluation': {
            'model': {
                'gpt_config': {
                    'model': 'text-ada-001'
                }
            }
        }
    }
    conf = config.update_config(user_config)
    model = llm.model_from_config(conf['evaluation']['model'])
    test_inputs = ["ßßβæçðñø"]
    # Get the indices of the word "equation" and "problem"

    log_probs, tokens = model.log_probs(test_inputs, [(2, 4)])
    reconstructed_tokens = ''.join(tokens[0]).strip()
    assert reconstructed_tokens == "βæ"
    assert len(log_probs) == 1
