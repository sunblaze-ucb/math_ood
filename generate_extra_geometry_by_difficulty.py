#!/usr/bin/env python3

"""
This file is so named because it generates the geometry problems that are used in places other than the explorative setting (those problems share a common framework, being based on a Euclidean plane construction framework).
"""
import json
import random
import os
import argparse
import yaml
from modules.generate_perpendicular_intersection_problems import (
    generate_dataset as polygon_chords_get_n_samples
)
from modules.generate_polygon_coloring_problems import (
    generate_n_constraint_problems as polygon_color_get_n_samples
)
from modules.generate_polygon_partition_problems import (
    generate_dataset as rotation_get_n_samples
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
        'polygon_chords',
        'rotation',
        'polygon_color',
    }
    
    # If problem_types not specified, use all available types
    if problem_types is None:
        problem_types = all_problem_types
    
    # Validate problem types
    problem_types = [pt for pt in problem_types if pt in all_problem_types]
    if not problem_types:
        print("No valid problem types specified")
        return
    
    # Container for all generated problems
    all_generated_problems = {}
    
    # Iterate through each problem type
    for problem_type in problem_types:
        print(f"\nGenerating {problem_type} problems at difficulty level {difficulty_level}...")
        
        # Skip if problem type not in config
        if config and 'levels' in config and problem_type not in config['levels']:
            print(f"Skipping {problem_type} as it's not defined in the config file")
            continue
            
        # Get level-specific parameters
        if config and 'levels' in config and problem_type in config['levels'] and difficulty_level in config['levels'][problem_type]:
            level_config = config['levels'][problem_type][difficulty_level]

        if problem_type == 'polygon_chords':
            problems = polygon_chords_get_n_samples(num_samples, level_config.get('min_num_diagonals', 5), level_config.get('max_num_diagonals', 13))
        elif problem_type == 'polygon_color':
            problems = polygon_color_get_n_samples(num_samples, level_config.get('min_vertices', 6), level_config.get('max_vertices', 12))
        elif problem_type == 'rotation':
            problems = rotation_get_n_samples(num_samples, level_config.get('min_num_diagonals', 1), level_config.get('max_num_diagonals', 4), level_config.get('min_vertices', 6), level_config.get('max_vertices', 12))
        
        # Save problems to a file
        filename = os.path.join(output_dir, f"geometry_{problem_type}_{difficulty_level}.jsonl")
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
    parser = argparse.ArgumentParser(description='Generate extra geometry problems at specific difficulty levels')
    
    parser.add_argument('--difficulty', type=str, required=True,
                        help='Difficulty level to generate (e.g., level_3)')
    parser.add_argument('--problem_types', type=str, nargs='+', default=None,
                        help='List of problem types to generate (default: all types)')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to generate (default: 20)')
    parser.add_argument('--output_dir', type=str, default="problems/specific_difficulty",
                        help='Output directory for problem files')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default="difficulty_configs/extra_geometry_config.yaml",
                        help='Path to YAML configuration file (default: difficulty_configs/extra_geometry_config.yaml)')
    
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