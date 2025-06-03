"""
Parse the ZebraLogic dataset from HF and convert it to a format matching the other problem generators.
Difficulty levels in this context are a simple mapping to the size of the grid.

This dataset is qualitatively different from all of the other ones in this repo, in that we didn't ourselves define the generation logic for the problems,
so that this script does not actually generate the problems. 
However, the problems are in fact produced by a generative process, so that this script can be extended to scale using the ZebraLogic paper's methods as desired.
Think of this particular family of problems as being a datapoint that demonstrates the principle, and credit goes to the ZebraLogic people for the generation logic.
"""
import argparse
import json
import datasets
import pdb
import random
import os

def get_n_samples(num_samples: int, grid_max_dim: int, start_idx: int = 0):
    dataset = datasets.load_dataset("WildEval/ZebraLogic", 'mc_mode')

    idx = start_idx
    problems = []
    while len(problems) < num_samples:
        try:
            problem = dataset['test'][idx]
        except IndexError:
            break
        idx += 1
        stats = {}
        size = problem['id'].split('-')[2]
        # the problem ids are of a regular format; the size is in the string and is represented as something like '3x3'
        # here a difficulty level of 4 may be assigned to a 4x4 grid, a 2x4 grid, etc.
        size1 = size[0]
        size2 = size[2]
        problem_difficulty = max(int(size1), int(size2))
        if problem_difficulty > grid_max_dim:
            continue

        stats['size'] = size
        choices = problem['choices']
        random.shuffle(choices)
        # Map choices to integers
        choice_mapping = {}
        for i, choice in enumerate(choices):
            choice_mapping[choice] = i

        # Find the answer's integer mapping
        answer_int = choice_mapping[problem['answer']]

        # Add the choice mapping to the end of the question
        choice_text = "\n\nChoices:"
        for i, choice in enumerate(choices):
            choice_text += f"\n{i}. {choice}"

        # Modify the question to include choices
        question_with_choices = problem['puzzle'] + choice_text
        question_with_choices += "\n\nWhen answering, please specify the number corresponding to the correct choice, instead of the choice itself."

        stats['question'] = question_with_choices
        stats['answer'] = answer_int
        stats['zebralogic_id'] = problem['id']
        problems.append(stats)
    print(f"Parsed {len(problems)} problems from ZebraLogic dataset")
    return problems, idx # need to use idx again later to avoid intersection between train and val, if they are parameterized the same