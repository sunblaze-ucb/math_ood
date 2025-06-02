#!/usr/bin/env python3

import yaml
import json
import argparse
import os
import random
from modules.number_theory import (
    generate_prime_mod_problem,
    generate_triple_count_problem, 
    generate_digit_sum_problem
)


def load_config(config_path):
    """
    Load YAML configuration from a file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return None
    
def save_problems_to_jsonl(problems, filename):
    """Save problems to a JSONL file.
    
    Args:
        problems: List of problem dictionaries
        filename: Path to the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write problems to a JSONL file
    with open(filename, 'w', encoding='utf-8') as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(problems)} problems to {filename}")

def generate_problems_with_difficulty(
    difficulty_level,
    problem_types=None,
    num_samples=10,
    output_dir="problems/specific_difficulty",
    config=None,
    seed=None
):
    """
    Generate number theory problems at a specific difficulty level.
    
    Args:
        difficulty_level: The specific difficulty level to generate (e.g., 'level_3')
        problem_types: List of problem types to generate (if None, generates all types)
        num_samples: Number of samples to generate
        output_dir: Directory to save output files
        config: Dictionary with configuration loaded from YAML
        seed: Random seed for reproducibility
    
    Saves results to jsonl files, one per problem type.
    """
    if seed is not None:
        random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define problem types and their generators
    all_problem_types = {
        'prime_mod': generate_prime_mod_problem,
        'triple_count': generate_triple_count_problem,
        'digit_sum': generate_digit_sum_problem
    }
    
    # If problem_types not specified, use all available types
    if problem_types is None:
        problem_types = list(all_problem_types.keys())
    
    # Validate problem types
    problem_types = [pt for pt in problem_types if pt in all_problem_types]
    if not problem_types:
        print("No valid problem types specified")
        return
    
    # Container for all generated problems
    all_generated_problems = {}
    
    # Iterate through each problem type
    for problem_type in problem_types:
        generator_func = all_problem_types[problem_type]
        print(f"\nGenerating {problem_type} problems at difficulty level {difficulty_level}...")
        
        # Skip if problem type not in config
        if config and 'levels' in config and problem_type not in config['levels']:
            print(f"Skipping {problem_type} as it's not defined in the config file") 
            continue
        
        # Get level-specific parameters
        if config and 'levels' in config and problem_type in config['levels'] and difficulty_level in config['levels'][problem_type]:
            level_config = config['levels'][problem_type][difficulty_level]
            difficulty = level_config.get('difficulty', 3)
            min_answer = level_config.get('min_answer', 0)
            max_answer = level_config.get('max_answer', 1000000000)
        else:
            # Default parameters if not specified in config
            difficulty = 3
            min_answer = 0
            max_answer = 1000
        
        count = 0
        attempts = 0
        max_attempts = num_samples * 100  # Limit attempts to avoid infinite loops
        
        problems = []
        
        while count < num_samples and attempts < max_attempts:
            print(attempts)
            attempts += 1
            try:

                # Generate problem with composition level from config
                problem = generator_func(difficulty)

                # Handle different answer types safely
                try:
                    answer_value = float(problem.answer)
                except (ValueError, TypeError):
                    # For non-numeric answers, skip constraints
                    answer_value = 0

                # Skip problems that don't meet constraints
                if problem.difficulty['difficulty'] < difficulty:
                    continue
                if min_answer >= 0 and answer_value < min_answer:
                    continue
                if answer_value > max_answer:
                    continue

                # Create problem entry
                problem_entry = {
                    "question": problem.question,
                    "answer": problem.answer,
                    "type": problem_type,
                    "difficulty": difficulty_level,
                    "parameters": {
                        "difficulty_level": difficulty_level,
                        "difficulty_info": problem.difficulty
                    }
                }

                problems.append(problem_entry)
                count += 1

                if count % 5 == 0 or count == num_samples:
                    print(f"Generated {count}/{num_samples} problems")
                    
            except Exception as e:
                print(f"Error generating {problem_type} problem: {e}")
                continue
        
        if attempts >= max_attempts and count < num_samples:
            print(f"Warning: Could only generate {count}/{num_samples} problems after {max_attempts} attempts")
        
        all_generated_problems[problem_type] = problems
        
        # Save problems to a file
        filename = os.path.join(output_dir, f"number_theory_{problem_type}_{difficulty_level}.jsonl")
        save_problems_to_jsonl(problems, filename)
    
    return all_generated_problems

def main():
    """Parse command line arguments and generate problems"""
    parser = argparse.ArgumentParser(description='Generate math problems with specific difficulty level')
    
    parser.add_argument('--difficulty', type=str, required=True,
                        help='Difficulty level to generate problems for')
    
    parser.add_argument('--problem_types', type=str, nargs='+', required=False,
                      choices=['prime_mod', 'triple_count', 'digit_sum'],
                      help='Problem types to generate (can specify multiple)')
    
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of problems to generate per problem type')
    
    parser.add_argument('--output_dir', type=str, default="problems/specific_difficulty",
                        help='Output file for generated problems (JSONL format)')
    
    parser.add_argument('--config', type=str, default="difficulty_configs/number_theory_config.yaml",
                        help='Path to YAML configuration file')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    if config is None:
        print(f"Failed to load configuration from {args.config}. Using default settings.")
    
    # Generate problems with specified parameters
    generate_problems_with_difficulty(
        difficulty_level=args.difficulty,
        problem_types=args.problem_types,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        config=config,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 
