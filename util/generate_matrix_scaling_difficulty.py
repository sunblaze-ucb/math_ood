#!/usr/bin/env python3

import json
import random
import os
import argparse
import re
import numpy as np
import yaml
from pathlib import Path
from modules.matrix_computations import (
    generate_matrix_inverse_problem,
    generate_matrix_multiplication_problem,
    generate_matrix_determinant_problem,
    generate_matrix_rank_problem,
    generate_eigenvalue_problem,
    generate_matrix_power_problem,
    generate_svd_decomposition_problem
)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_config(config_path="difficulty_configs/matrix_config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_scaling_difficulty_problems(
    config=None,
    min_level=None,
    max_level=None,
    samples_per_level=None,
    output_dir="problems/difficulty",
    seed=None,
    problem_types=None
):
    """
    Generate matrix problems with scaling difficulty levels.
    
    Args:
        config: Configuration dictionary loaded from YAML
        min_level: Minimum complexity level to start from (overrides config)
        max_level: Maximum complexity level to reach (overrides config)
        samples_per_level: Number of samples per complexity level (overrides config)
        output_dir: Directory to save output files
        seed: Random seed for reproducibility
        problem_types: List of problem types to generate (if None, generate all types)
    
    Saves results to jsonl files, one per problem type, with problems organized
    by increasing complexity level.
    """
    if config is None:
        config = load_config()
    
    # Use default values from config if not specified
    if min_level is None:
        min_level = config.get('defaults', {}).get('min_level', 1)
    if max_level is None:
        max_level = config.get('defaults', {}).get('max_level', 7)
    if samples_per_level is None:
        samples_per_level = config.get('defaults', {}).get('samples_per_level', 100)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Define problem types and their generators
    problem_generators = {
        'inverse': generate_matrix_inverse_problem,
        'multiplication': generate_matrix_multiplication_problem,
        'determinant': generate_matrix_determinant_problem,
        'rank': generate_matrix_rank_problem,
        'eigenvalues': generate_eigenvalue_problem,
        'power': generate_matrix_power_problem,
        'svd': generate_svd_decomposition_problem
    }
    
    # Filter problem types if specified
    if problem_types:
        problem_generators = {k: v for k, v in problem_generators.items() if k in problem_types}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each problem type
    for problem_type, generator_func in problem_generators.items():
        print(f"\nGenerating scaling difficulty problems for {problem_type}...")
        
        # Container for all problems
        all_problems = []
        
        # Generate problems for each complexity level
        for level in range(min_level, max_level + 1):
            level_key = f"level_{level}"
            
            # Skip if level not defined in config for this problem type
            if level_key not in config['levels'].get(problem_type, {}):
                print(f"Skipping level {level} for {problem_type} as it's not defined in config")
                continue
                
            print(f"Generating {samples_per_level} {problem_type} problems at complexity level {level}...")
            
            # Get parameters for this level from config
            level_params = config['levels'][problem_type][level_key]
            
            count = 0
            attempts = 0
            max_attempts = samples_per_level * 10  # Limit attempts to avoid infinite loops
            
            level_problems = []
            
            while count < samples_per_level and attempts < max_attempts:
                attempts += 1
                try:
                    # Generate problem with parameters from config based on problem type
                    if problem_type in ['inverse', 'determinant', 'eigenvalues']:
                        # Choose a random matrix type from available options
                        special_type = random.choice(level_params.get('matrix_types', [None]))
                        problem = generator_func(
                            rows=level_params['rows'],
                            special_type=special_type,
                            min_val=level_params['min_val'],
                            max_val=level_params['max_val']
                        )
                    elif problem_type == 'power':
                        special_type = random.choice(level_params.get('matrix_types', [None]))
                        problem = generator_func(
                            rows=level_params['rows'],
                            special_type=special_type,
                            min_val=level_params['min_val'],
                            max_val=level_params['max_val'],
                            min_power=level_params.get('min_power', 2),
                            max_power=level_params.get('max_power', 4)
                        )
                    elif problem_type == 'multiplication':
                        # For multiplication, use min_dim and max_dim to determine matrix sizes
                        min_dim = level_params['min_dim']
                        max_dim = level_params['max_dim']
                        rows_a = random.randint(min_dim, max_dim)
                        cols_a = random.randint(min_dim, max_dim)
                        cols_b = random.randint(min_dim, max_dim)
                        problem = generator_func(
                            rows_a=rows_a,
                            cols_a=cols_a,
                            cols_b=cols_b,
                            min_val=level_params['min_val'],
                            max_val=level_params['max_val']
                        )
                    else:  # rank and svd
                        # For rectangular matrix problems
                        rows = level_params['rows']
                        cols = level_params['cols']
                        if problem_type == 'rank':
                            specific_rank = random.randint(1, min(rows, cols) - 1) if min(rows, cols) > 1 else 1
                            problem = generator_func(
                                rows=rows,
                                cols=cols,
                                min_val=level_params['min_val'],
                                max_val=level_params['max_val'],
                                specific_rank=specific_rank
                            )
                        else:  # svd
                            problem = generator_func(
                                rows=rows,
                                cols=cols,
                                min_val=level_params['min_val'],
                                max_val=level_params['max_val']
                            )
                    
                    # Calculate difficulty level
                    difficulty = level  # Base difficulty from complexity level
                    
                    # Create parameter dict based on problem type
                    if problem_type in ['inverse', 'determinant', 'eigenvalues']:
                        parameters = {
                            "rows": level_params['rows'],
                            "cols": level_params['rows'],
                            "level": level,
                            "min_val": level_params['min_val'],
                            "max_val": level_params['max_val'],
                            "val_scale": 6 + (2 * level),
                            "matrix_type": special_type
                        }
                    elif problem_type == 'power':
                        parameters = {
                            "rows": level_params['rows'],
                            "cols": level_params['rows'],
                            "level": level,
                            "min_val": level_params['min_val'],
                            "max_val": level_params['max_val'],
                            "min_power": level_params.get('min_power', 2),
                            "max_power": level_params.get('max_power', 4),
                            "val_scale": 6 + (2 * level),
                            "matrix_type": special_type
                        }
                    elif problem_type == 'multiplication':
                        parameters = {
                            "rows_a": rows_a,
                            "cols_a": cols_a,
                            "rows_b": cols_a,  # For matrix multiplication, rows_b must equal cols_a
                            "cols_b": cols_b,
                            "level": level,
                            "min_val": level_params['min_val'],
                            "max_val": level_params['max_val'],
                            "val_scale": 6 + (2 * level)
                        }
                    else:  # rank and svd
                        parameters = {
                            "rows": level_params['rows'],
                            "cols": level_params['cols'],
                            "level": level,
                            "min_val": level_params['min_val'],
                            "max_val": level_params['max_val'],
                            "val_scale": 6 + (2 * level)
                        }
                        if problem_type == 'rank':
                            parameters["specific_rank"] = specific_rank
                    
                    # Extract matrix from the problem text
                    matrix_str = re.search(r'\[((\[.*?\])(,\n \[.*?\])*)\]', problem.question)
                    matrix_text = matrix_str.group(0) if matrix_str else "Matrix extraction failed"
                    
                    # Create problem entry
                    problem_entry = {
                        "question": problem.question,
                        "answer": problem.answer,
                        "type": problem_type,
                        "difficulty": difficulty,
                        "parameters": parameters,
                        "matrix_text": matrix_text
                    }
                    
                    level_problems.append(problem_entry)
                    count += 1
                    
                    if count % 5 == 0 or count == samples_per_level:
                        print(f"Generated {count}/{samples_per_level} problems at level {level}")
                    
                except Exception as e:
                    print(f"Error generating {problem_type} problem at level {level}: {e}")
                    continue
            
            if attempts >= max_attempts and count < samples_per_level:
                print(f"Warning: Could only generate {count}/{samples_per_level} problems at level {level} after {max_attempts} attempts")
            
            # Add problems from this level to the overall collection
            all_problems.extend(level_problems)
        
        # Save all problems for this type to a file
        filename = os.path.join(output_dir, f"arithmetic_matrix_{problem_type}_scaling_difficulty.jsonl")
        save_problems_to_jsonl(all_problems, filename)
        
        # Also create a summary file with statistics
        generate_difficulty_summary(all_problems, problem_type, output_dir)
    
    print(f"\nCompleted generating scaling difficulty problems for all types")

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
            f.write(json.dumps(problem, cls=NumpyEncoder, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(problems)} problems to {filename}")

def generate_difficulty_summary(problems, problem_type, output_dir):
    """Generate a summary of problems by difficulty level.
    
    Args:
        problems: List of problem dictionaries
        problem_type: Type of problem
        output_dir: Directory to save the summary file
    """
    # Group problems by level
    problems_by_level = {}
    for problem in problems:
        level = problem["parameters"]["level"]
        if level not in problems_by_level:
            problems_by_level[level] = []
        problems_by_level[level].append(problem)
    
    # Calculate statistics for each level
    summary = []
    for level, level_problems in sorted(problems_by_level.items()):
        difficulties = [p["difficulty"] for p in level_problems]
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        
        summary.append({
            "problem_type": problem_type,
            "complexity_level": level,
            "num_problems": len(level_problems),
            "avg_difficulty": avg_difficulty,
            "min_difficulty": min(difficulties) if difficulties else 0,
            "max_difficulty": max(difficulties) if difficulties else 0
        })
    
    # Save summary to a JSON file
    summary_filename = os.path.join(output_dir, f"arithmetic_matrix_{problem_type}_difficulty.json")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved difficulty summary to {summary_filename}")

def main():
    """Parse command line arguments and generate problems"""
    parser = argparse.ArgumentParser(description='Generate matrix problems with scaling difficulty')
    
    parser.add_argument('--config', type=str, default="difficulty_configs/matrix_config.yaml",
                        help='Path to YAML configuration file')
    parser.add_argument('--min_level', type=int, default=None,
                        help='Minimum complexity level (overrides config file)')
    parser.add_argument('--max_level', type=int, default=None,
                        help='Maximum complexity level (overrides config file)')
    parser.add_argument('--samples_per_level', type=int, default=None,
                        help='Number of samples per complexity level (overrides config file)')
    parser.add_argument('--output_dir', type=str, default="problems/difficulty",
                        help='Output directory for problem files')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--types', type=str, nargs='+', 
                      choices=['inverse', 'multiplication', 'determinant', 'rank', 
                               'eigenvalues', 'power', 'svd', 'all'],
                      default=['all'],
                      help='Problem types to generate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Convert 'all' to all problem types
    if 'all' in args.types:
        problem_types = ['inverse', 'multiplication', 'determinant', 'rank', 
                       'eigenvalues', 'power', 'svd']
    else:
        problem_types = args.types
    
    # Generate problems with specified parameters
    generate_scaling_difficulty_problems(
        config=config,
        min_level=args.min_level,
        max_level=args.max_level,
        samples_per_level=args.samples_per_level,
        output_dir=args.output_dir,
        seed=args.seed,
        problem_types=problem_types
    )

if __name__ == "__main__":
    main() 