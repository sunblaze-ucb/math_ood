#!/usr/bin/env python3

import json
import random
import os
import argparse
import logging
import yaml
from generate_basics_scaling_difficulty import (
    load_difficulty_config,
    configure_generator,
    get_difficulty_info,
    save_problems_to_jsonl
)
from modules.algebra import _solve_linear_system, polynomial_roots
from modules.arithmetic import mixed
from modules.number_probs import list_prime_factors, gcd

def generate_problems_at_difficulty(
    difficulty_level,
    problem_types=None,
    num_samples=50,
    output_dir="problems/specific_difficulty",
    seed=None,
    config_path="difficulty_configs/basics_config.yaml"
):
    """
    Generate math problems at a specific difficulty level.
    
    Args:
        difficulty_level: The difficulty level to generate problems for (1-5)
        problem_types: List of problem types to generate, or None for all
        num_samples: Number of samples to generate
        output_dir: Directory to save output files
        seed: Random seed for reproducibility
        config_path: Path to the difficulty configuration file
    
    Saves results to separate JSONL files for each problem type with the naming convention:
    {problem_type}_level_{difficulty_level}.jsonl
    """
    if seed is not None:
        random.seed(seed)
    
    # Load difficulty configuration
    difficulty_config = load_difficulty_config(config_path)
    
    # Define generators for each problem type
    problem_generators = {
        'algebra_linear_equation': {
            'generator': _solve_linear_system
        },
        'algebra_polynomial_roots': {
            'generator': polynomial_roots
        },
        'arithmetic_mixed': {
            'generator': mixed
        },
        'arithmetic_list_prime_factors': {
            'generator': list_prime_factors
        },
        'arithmetic_gcd': {
            'generator': gcd
        }
    }
    
    # Filter problem types if specified
    if problem_types:
        problem_generators = {k: v for k, v in problem_generators.items() if k in problem_types}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # Iterate through each problem type
    for problem_type, generator_info in problem_generators.items():
        logging.info(f"Generating {num_samples} {problem_type} problems at difficulty level {difficulty_level}...")
        
        # Configure generator based on problem type and difficulty level
        generator = configure_generator(problem_type, difficulty_level, generator_info['generator'], difficulty_config)
        
        count = 0
        attempts = 0
        max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops
        
        type_problems = []
        
        while count < num_samples and attempts < max_attempts:
            attempts += 1
            try:
                # Generate problem
                problem = generator()

                # Create problem entry
                problem_entry = {
                    "question": str(problem.question),
                    "answer": str(problem.answer),
                    "type": problem_type,
                    "difficulty_level": difficulty_level,
                    "parameters": {
                        "difficulty_info": get_difficulty_info(problem_type, difficulty_level)
                    }
                }

                type_problems.append(problem_entry)
                count += 1

                if count % 10 == 0 or count == num_samples:
                    logging.info(f"Generated {count}/{num_samples} problems of type {problem_type}")
                
            except Exception as e:
                error_msg = str(e)
                if "No valid samplers found" in error_msg:
                    # Just continue with next attempt for this specific error
                    logging.debug(f"Skipping problematic polynomial: {error_msg}")
                    continue
                else:
                    # Log other errors
                    logging.error(f"Error generating {problem_type} problem: {e}")
                    continue
        
        if attempts >= max_attempts and count < num_samples:
            logging.warning(f"Warning: Could only generate {count}/{num_samples} problems of type {problem_type} after {max_attempts} attempts")
        
        # Save problems from this type to a separate file with expected naming convention
        filename = os.path.join(output_dir, f"{problem_type}_level_{difficulty_level}.jsonl")
        save_problems_to_jsonl(type_problems, filename)
        generated_files.append(filename)
        
        logging.info(f"Saved {len(type_problems)} {problem_type} problems to {filename}")
    
    logging.info(f"\nCompleted generating problems for all types at difficulty level {difficulty_level}")
    return generated_files

def main():
    """Parse command line arguments and generate problems."""
    parser = argparse.ArgumentParser(description='Generate math problems at a specific difficulty level')
    
    parser.add_argument('--difficulty', type=int, required=True,
                      help='Difficulty level (1-5)')
    parser.add_argument('--problem_types', type=str, nargs='+', default=None,
                      help='Specific problem types to generate (default: all)')
    parser.add_argument('--num_samples', type=int, default=50,
                      help='Number of samples to generate per problem type (default: 50)')
    parser.add_argument('--output_dir', type=str, default="problems/specific_difficulty",
                      help='Output directory for problem files')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--config_path', type=str, default="difficulty_configs/basics_config.yaml",
                      help='Path to the difficulty configuration file')
    
    args = parser.parse_args()
    
    # Validate difficulty level
    if args.difficulty < 1 or args.difficulty > 5:
        parser.error("Difficulty level must be between 1 and 5")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Generate problems with specified parameters
    output_files = generate_problems_at_difficulty(
        difficulty_level=args.difficulty,
        problem_types=args.problem_types,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        config_path=args.config_path
    )
    
    print(f"Problems generated successfully and saved to:")
    for file in output_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main() 