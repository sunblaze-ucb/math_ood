#!/usr/bin/env python3

import json
import argparse
import os
from util.generate_matrix_scaling_difficulty import (
    load_config,
    NumpyEncoder,
    generate_matrix_inverse_problem,
    generate_matrix_multiplication_problem,
    generate_matrix_determinant_problem,
    generate_matrix_rank_problem,
    generate_eigenvalue_problem,
    generate_matrix_power_problem,
    generate_svd_decomposition_problem
)
import random
import re
import numpy as np

def generate_specific_difficulty_problems(
    problem_type,
    difficulty_level,
    num_samples=10,
    config_path="difficulty_configs/matrix_config.yaml",
    output_file=None,
    seed=None
):
    """
    Generate problem samples at a specific difficulty level.
    
    Args:
        problem_type: Type of problem to generate ('inverse', 'multiplication', etc.)
        difficulty_level: Specific difficulty level to generate problems for
        num_samples: Number of problems to generate
        config_path: Path to the YAML configuration file
        output_file: Path to save the generated problems (if None, problems are returned)
        seed: Random seed for reproducibility
        
    Returns:
        List of generated problems if output_file is None, otherwise None
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load configuration
    config = load_config(config_path)
    
    # Check if the requested difficulty level exists in the config
    level_key = f"level_{difficulty_level}"
    if level_key not in config['levels'].get(problem_type, {}):
        raise ValueError(f"Difficulty level {difficulty_level} not defined for {problem_type} in the config")
    
    # Get parameters for this level from config
    level_params = config['levels'][problem_type][level_key]
    
    # Define problem generators
    problem_generators = {
        'inverse': generate_matrix_inverse_problem,
        'multiplication': generate_matrix_multiplication_problem,
        'determinant': generate_matrix_determinant_problem,
        'rank': generate_matrix_rank_problem,
        'eigenvalues': generate_eigenvalue_problem,
        'power': generate_matrix_power_problem,
        'svd': generate_svd_decomposition_problem
    }
    
    if problem_type not in problem_generators:
        raise ValueError(f"Problem type '{problem_type}' not supported. Choose from: {', '.join(problem_generators.keys())}")
    
    generator_func = problem_generators[problem_type]
    
    # Generate problems
    problems = []
    count = 0
    attempts = 0
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops
    
    print(f"Generating {num_samples} {problem_type} problems at difficulty level {difficulty_level}...")
    
    while count < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            # Generate problem based on problem type and parameters
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
            
            # Create parameter dict based on problem type
            if problem_type in ['inverse', 'determinant', 'eigenvalues']:
                parameters = {
                    "rows": level_params['rows'],
                    "cols": level_params['rows'],
                    "level": difficulty_level,
                    "min_val": level_params['min_val'],
                    "max_val": level_params['max_val'],
                    "val_scale": 6 + (2 * difficulty_level),
                    "matrix_type": special_type
                }
            elif problem_type == 'power':
                parameters = {
                    "rows": level_params['rows'],
                    "cols": level_params['rows'],
                    "level": difficulty_level,
                    "min_val": level_params['min_val'],
                    "max_val": level_params['max_val'],
                    "min_power": level_params.get('min_power', 2),
                    "max_power": level_params.get('max_power', 4),
                    "val_scale": 6 + (2 * difficulty_level),
                    "matrix_type": special_type
                }
            elif problem_type == 'multiplication':
                parameters = {
                    "rows_a": rows_a,
                    "cols_a": cols_a,
                    "rows_b": cols_a,  # For matrix multiplication, rows_b must equal cols_a
                    "cols_b": cols_b,
                    "level": difficulty_level,
                    "min_val": level_params['min_val'],
                    "max_val": level_params['max_val'],
                    "val_scale": 6 + (2 * difficulty_level)
                }
            else:  # rank and svd
                parameters = {
                    "rows": level_params['rows'],
                    "cols": level_params['cols'],
                    "level": difficulty_level,
                    "min_val": level_params['min_val'],
                    "max_val": level_params['max_val'],
                    "val_scale": 6 + (2 * difficulty_level)
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
                "difficulty": difficulty_level,
                "parameters": parameters,
                "matrix_text": matrix_text
            }
            
            problems.append(problem_entry)
            count += 1
            
            if count % 5 == 0 or count == num_samples:
                print(f"Generated {count}/{num_samples} problems")
            
        except Exception as e:
            print(f"Error generating {problem_type} problem at level {difficulty_level}: {e}")
            continue
    
    if attempts >= max_attempts and count < num_samples:
        print(f"Warning: Could only generate {count}/{num_samples} problems after {max_attempts} attempts")
    
    # Save problems to a file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(json.dumps(problem, cls=NumpyEncoder, ensure_ascii=False) + '\n')
        print(f"Saved {len(problems)} problems to {output_file}")
    
    return problems

def main():
    """Parse command line arguments and generate problems"""
    parser = argparse.ArgumentParser(description='Generate math problems with specific difficulty level')
    
    parser.add_argument('--problem_types', type=str, nargs='+', required=True,
                      choices=['inverse', 'multiplication', 'determinant', 'rank', 
                               'eigenvalues', 'power', 'svd'],
                      help='Problem types to generate (can specify multiple)')
    
    parser.add_argument('--difficulty', type=int, required=True,
                        help='Difficulty level to generate problems for')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of problems to generate per problem type')
    
    parser.add_argument('--config', type=str, default="difficulty_configs/matrix_config.yaml",
                        help='Path to YAML configuration file')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for generated problems (JSONL format)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate problems for each problem type
    all_problems = []
    for problem_type in args.problem_types:
        print(f"\n=== Generating {problem_type} problems ===")
        
        # Generate output filename for this problem type if base output is specified
        if args.output and len(args.problem_types) == 1:
            output_file = args.output
        else:
            output_file = f"problems/specific_difficulty/arithmetic_matrix_{problem_type}_level_{args.difficulty}.jsonl"
        problems = generate_specific_difficulty_problems(
            problem_type=problem_type,
            difficulty_level=args.difficulty,
            num_samples=args.num_samples,
            config_path=args.config,
            output_file=output_file,
            seed=args.seed
        )
        
        all_problems.extend(problems)
    
    print(f"\nTotal problems generated: {len(all_problems)}")

if __name__ == "__main__":
    main() 