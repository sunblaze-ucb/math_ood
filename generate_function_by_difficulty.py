#!/usr/bin/env python3

import json
import random
import os
import argparse
import yaml
from modules.function_analysis import (
    generate_intersection_problem,
    generate_intersection_coordinates_problem,
    generate_extrema_problem,
    generate_extrema_coordinates_problem,
    generate_area_problem,
    generate_zero_problem,
    generate_derivative_sign_problem
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

def generate_problems_with_difficulty(
    difficulty_level,
    problem_types=None,
    num_samples=20,
    output_dir="problems",
    config=None,
    seed=None
):
    """
    Generate function analysis problems at a specific difficulty level.
    
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
        'intersection': generate_intersection_problem,
        'intersection_coords': generate_intersection_coordinates_problem,
        'extrema': generate_extrema_problem,
        'extrema_coords': generate_extrema_coordinates_problem,
        'area': generate_area_problem,
        'zeros': generate_zero_problem,
        'derivative_sign': generate_derivative_sign_problem
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
            composition_nums = level_config.get('composition_nums', 2)
            min_difficulty = level_config.get('difficulty', 0)
            min_answer = level_config.get('min_answer', 0)
            max_answer = level_config.get('max_answer', float('inf'))
        else:
            # Default parameters if not specified in config
            composition_nums = int(difficulty_level.split('_')[1]) + 1  # level_1 -> 2, level_2 -> 3, etc.
            min_difficulty = 0
            min_answer = 0
            max_answer = float('inf')
        
        count = 0
        attempts = 0
        max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops
        
        problems = []
        
        while count < num_samples and attempts < max_attempts:
            attempts += 1
            try:
                # Generate problem with composition level from config
                problem = generator_func(composition_nums=composition_nums)

                # Get difficulty level
                difficulty = problem.difficulty.get('level', 0) if problem.difficulty else 0

                # Handle different answer types safely
                try:
                    answer_value = float(problem.answer)
                except (ValueError, TypeError):
                    # For non-numeric answers, skip constraints
                    answer_value = 0

                # Skip problems that don't meet constraints
                if difficulty < min_difficulty:
                    continue
                if min_answer > 0 and answer_value <= min_answer:
                    continue
                if answer_value > max_answer:
                    continue

                # Create problem entry
                problem_entry = {
                    "question": problem.question,
                    "answer": problem.answer,
                    "type": problem_type,
                    "difficulty": difficulty,
                    "parameters": {
                        "composition_level": composition_nums,
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
        filename = os.path.join(output_dir, f"algebra_func_{problem_type}_{difficulty_level}.jsonl")
        save_problems_to_jsonl(problems, filename)
    
    return all_generated_problems

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

def main():
    """Parse command line arguments and generate problems"""
    parser = argparse.ArgumentParser(description='Generate function analysis problems at specific difficulty levels')
    
    parser.add_argument('--difficulty', type=str, required=True,
                        help='Difficulty level to generate (e.g., level_3)')
    parser.add_argument('--problem_types', type=str, nargs='+', default=None,
                        help='List of problem types to generate (default: all types)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate (default: 20)')
    parser.add_argument('--output_dir', type=str, default="problems/specific_difficulty",
                        help='Output directory for problem files')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default="difficulty_configs/func_config.yaml",
                        help='Path to YAML configuration file (default: difficulty_configs/func_config.yaml)')
    
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