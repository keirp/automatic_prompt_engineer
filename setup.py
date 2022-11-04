from setuptools import setup

setup(
    name='automatic_prompt_engineer',
    version='1.0',
    description='',
    author='Keiran Paster',
    author_email='keirp@cs.toronto.edu',
    packages=['automatic_prompt_engineer',
              'automatic_prompt_engineer.evaluation'],
    package_data={'automatic_prompt_engineer': ['configs/*']},
    install_requires=[
        'numpy',
        'openai',
        'fire',
        'tqdm',
        'gradio',
    ],
)
