from automatic_prompt_engineer import generate, config, template


def test_generate_instruction():
    prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    prompt_gen_template = template.GenerationTemplate(prompt_gen_template)
    demos_template = template.DemosTemplate(demos_template)

    inputs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    outputs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    data = inputs, outputs

    user_config = {
        'generation': {
            'num_subsamples': 2,
            'num_demos': 5,
            'num_prompts_per_subsample': 2,
            'model': {
                'gpt_config': {
                    'model': 'text-ada-001'
                }
            }
        }
    }

    conf = config.update_config(user_config)

    prompts = generate.generate_prompts(
        prompt_gen_template, demos_template, data, conf['generation'])
    assert len(prompts) == 4
