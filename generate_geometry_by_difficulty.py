#!/usr/bin/env python3

import pdb
import json
import argparse
import os
import random
import re
import numpy as np
import yaml
from modules.geometry.pipeline import main as pipeline_main, make_parser

def generate_specific_difficulty_problems(
    problem_type,
    difficulty_level,
    num_samples=100,
    config_path="difficulty_configs/geometry_config.yaml",
    output_file=None,
    seed=None
):


    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    level_key = f"level_{difficulty_level}"
    if level_key not in config['levels'].get(problem_type, {}):
        raise ValueError(f"Difficulty level {difficulty_level} not defined for {problem_type} in the config")
    
    # Get parameters for this level from config
    level_params = config['levels'][problem_type][level_key]
    min_length = level_params['min_length']
    max_length = level_params['max_length']

    parser = make_parser()

    # some heuristics, to run the right number of sampling jobs 
    sample_multipliers = [500, 800, 1000, 2000, 3000]
    count = sample_multipliers[difficulty_level - 1] * num_samples
    num_generator_commands_per_level = [100, 100, 100, 150, 200]
    num_generator_commands = num_generator_commands_per_level[difficulty_level - 1]
    args = parser.parse_args(['--count', str(count), '--num_generator_commands', str(num_generator_commands), '--generator_command_types', problem_type, '--max_workers', '64',
                              '--output_translations_dir', 'modules/geometry/natural_language_problems',
                              '--min_num_generator_commands', str(min_length),
                              '--max_num_generator_commands', str(max_length),
                              '--return_problems'])
    problems = pipeline_main(args)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join([json.dumps(problem) for problem in problems]))

    print (f"Generated {len(problems)} problems for {problem_type} at difficulty level {difficulty_level} (may not be the exact amount you wanted, due to rejection sampling)")
    return problems


def main():
    """Parse command line arguments and generate problems"""
    parser = argparse.ArgumentParser(description='Generate math problems with specific difficulty level')
    
    parser.add_argument('--problem_types', type=str, required=True,
                      choices=['polygon', 'circle', 'triangle', 'basic'],
                      help='Problem type to generate')
    
    parser.add_argument('--difficulty', type=int, required=True,
                        help='Difficulty level to generate problems for')
    
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of problems to generate')
    
    parser.add_argument('--config', type=str, default="difficulty_configs/geometry_config.yaml",
                        help='Path to YAML configuration file')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for generated problems (JSONL format)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    

    args.output = f"problems/specific_difficulty/geometry_{args.problem_types}_level_{args.difficulty}.jsonl"

    generate_specific_difficulty_problems(
        problem_type=args.problem_types,
        difficulty_level=args.difficulty,
        num_samples=args.num_samples,
        config_path=args.config,
        output_file=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 