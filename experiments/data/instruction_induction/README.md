# Data

Data for the instruction induction experiments.

## Content

- annotations: contains the gold annotations, used for reference-based evaluation.
- induction_input: contains the inputs used in our instruction induction experiments. 
	Each input is composed of our instruction induction prompt and five task input-output demonstrations.
- raw:
	- induce: contains input-output demonstrations that were used to construct the instruction induction inputs.
	- execute: held out examples for the execution accuracy evaluation metric.