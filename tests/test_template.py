from automatic_prompt_engineer import template


def test_generate_lets():
    temp = "Q: [INPUT] A: Let's [APE]. [OUTPUT]"
    generation_template = template.GenerationTemplate(temp)
    infilled = generation_template.fill(
        full_demo="test str",
        input="test input",
        output="test output",
    )
    assert infilled == "Q: test input A: Let's [APE]. test output"


def test_generate_instruction():
    temp = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    generation_template = template.GenerationTemplate(temp)
    infilled = generation_template.fill(
        full_demo="test str",
        input="test input",
        output="test output",
    )
    assert infilled == "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\ntest str\n\nThe instruction was to [APE]"


def test_eval_lets():
    temp = "Q: [INPUT] A: Let's [PROMPT]. [OUTPUT]"
    eval_template = template.EvalTemplate(temp)
    infilled = eval_template.fill(
        prompt="test prompt",
        full_demo="test str",
        input="test input",
        output="test output",
    )
    assert infilled == "Q: test input A: Let's test prompt. test output"


def test_eval_instruction():
    temp = "[full_DEMO]\n\nInstruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]"
    eval_template = template.EvalTemplate(temp)
    infilled = eval_template.fill(
        prompt="test prompt",
        full_demo="test str",
        input="test input",
        output="test output",
    )
    assert infilled == "test str\n\nInstruction: test prompt\nInput: test input\nOutput: test output"


def test_demo_template():
    temp = "Input: [INPUT]\nOutput: [OUTPUT]"
    demo_template = template.DemosTemplate(temp, delimiter="\n")
    data = ["a", "b", "c"], ["1", "2", "3"]
    infilled = demo_template.fill(
        data
    )
    assert infilled == "Input: a\nOutput: 1\nInput: b\nOutput: 2\nInput: c\nOutput: 3"
