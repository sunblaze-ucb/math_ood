#!/usr/bin/env python3

import json
import os
import argparse
import yaml
from modules.combinatories import (
    generate_probability_problem,
    distribute_letter_problem,
    generate_distribution_problem,
    find_letter_distributions,
    generate_pattern_matching_problem
)

def load_config(config_path="difficulty_configs/combi_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_distribution_problems(difficulty_level, num_samples, output_dir, config):
    """Generate distribution problems at a specific difficulty level."""
    problems = []
    
    print(f"Generating {num_samples} distribution problems at difficulty level {difficulty_level}...")
    
    count = 0
    attempts = 0
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops
    
    while count < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            # Get parameters from config for this level
            level_key = f"level_{difficulty_level}"
            
            if level_key in config["levels"]["distribution"]:
                dist_config = config["levels"]["distribution"][level_key]
                num_letter_types = dist_config["num_letter_types_range"][0]
                num_boxes = dist_config["num_boxes_range"][0]
                total_letters = dist_config["total_letters_range"][0]
            else:
                # Fall back to calculated parameters if level not in config
                num_letter_types = min(2 + difficulty_level // 2, 5)  # 2 to 5 based on level
                num_boxes = min(2 + difficulty_level // 2, 5)  # 2 to 5 based on level
                total_letters = 4 + difficulty_level * 2  # Scales with level
            
            labeled_boxes = True if difficulty_level % 2 == 0 else False  # Alternate between labeled and unlabeled
            
            # Generate the problem specification
            problem_spec = generate_distribution_problem(
                num_letter_types=num_letter_types,
                num_boxes=num_boxes,
                total_letters=total_letters,
                labeled_boxes=labeled_boxes
            )
            
            # Find all distributions
            distributions, count_result = find_letter_distributions(
                problem_spec["letters"],
                problem_spec["box_sizes"],
                problem_spec["labeled_boxes"]
            )
            
            # Generate the problem
            problem = distribute_letter_problem(
                num_letter_type_range=[num_letter_types],
                num_boxes_range=[num_boxes],
                total_letters_range=[total_letters]
            )
            
            # Create problem entry
            problem_entry = {
                "question": problem.question,
                "answer": problem.answer,
                "type": "distribution",
                "difficulty": difficulty_level,
                "parameters": {
                    "num_letter_types": num_letter_types,
                    "num_boxes": num_boxes,
                    "total_letters": total_letters,
                    "labeled_boxes": labeled_boxes,
                    "letters": problem_spec["letters"],
                    "box_sizes": problem_spec["box_sizes"]
                }
            }
            
            problems.append(problem_entry)
            count += 1
            
            if count % 1 == 0 or count == num_samples:
                print(f"Generated {count}/{num_samples} problems")
            
        except Exception as e:
            continue
    
    if attempts >= max_attempts and count < num_samples:
        print(f"Warning: Could only generate {count}/{num_samples} problems after {max_attempts} attempts")
    
    return problems

def generate_pattern_matching_problems(difficulty_level, num_samples, output_dir, config):
    """Generate pattern matching problems at a specific difficulty level."""
    problems = []
    
    print(f"Generating {num_samples} pattern matching problems at difficulty level {difficulty_level}...")
    
    count = 0
    attempts = 0
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops
    
    while count < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            # Get parameters from config for this level
            level_key = f"level_{difficulty_level}"
            
            if level_key in config["levels"]["pattern_matching"]:
                pattern_config = config["levels"]["pattern_matching"][level_key]
                length = pattern_config["length_range"][0]
                total_letters = pattern_config["total_letters_range"][0]
            else:
                # Fall back to calculated parameters if level not in config
                length = min(2 + difficulty_level // 2, 6)  # 2 to 6 based on level
                total_letters = 4 + difficulty_level * 2  # Scales with level
            
            # Generate the problem
            problem_statement, words, pattern_str, expected_matches, rounded_expected = generate_pattern_matching_problem(
                length=length,
                total_letters=total_letters,
                debug=False
            )
            
            # Create problem entry
            problem_entry = {
                "question": problem_statement,
                "answer": rounded_expected,
                "type": "pattern_matching",
                "difficulty": difficulty_level,
                "parameters": {
                    "length": length,
                    "total_letters": total_letters,
                    "pattern": pattern_str,
                    "expected_matches": float(expected_matches)
                }
            }
            
            problems.append(problem_entry)
            count += 1
            
            if count % 5 == 0 or count == num_samples:
                print(f"Generated {count}/{num_samples} problems")
            
        except Exception as e:
            continue
    
    if attempts >= max_attempts and count < num_samples:
        print(f"Warning: Could only generate {count}/{num_samples} problems after {max_attempts} attempts")
    
    return problems

def generate_probability_problems(difficulty_level, num_samples, output_dir, config, event_type=None):
    """Generate probability problems at a specific difficulty level.
    
    Args:
        difficulty_level: The difficulty level for generated problems
        num_samples: Number of samples to generate
        output_dir: Directory to save output files
        config: Configuration dictionary
        event_type: Specific event type to generate, if None generates problems with a mix of event types
    """
    problems = []
    
    # Event types for probability problems
    event_types = config.get('event_types', [
        'no_fixed_points',
        'no_specific_letter_fixed',
        'exactly_n_specific_fixed',
        'at_least_n_specific_fixed'
    ])
    
    if event_type and event_type not in event_types:
        print(f"Warning: Event type '{event_type}' not recognized. Using random event types.")
        event_type = None
    
    print(f"Generating {num_samples} probability problems at difficulty level {difficulty_level}" + 
          (f" with event type '{event_type}'" if event_type else " with mixed event types"))
    
    count = 0
    attempts = 0
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops
    
    while count < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            # Get parameters from config for this level
            level_key = f"level_{difficulty_level}"
            
            if level_key in config["levels"]["probability"]:
                prob_config = config["levels"]["probability"][level_key]
                length = prob_config["length_range"][0]
                total_letters = prob_config["total_letters_range"][0]
            else:
                # Fall back to calculated parameters if level not in config
                length = 3 + int(difficulty_level * 1.2)  # 3 + level^1.2
                total_letters = 4 + difficulty_level * 2  # Scales with level
            
            letter_count = max(3, difficulty_level)
            
            # Select event type if not specified
            current_event_type = event_type if event_type else event_types[count % len(event_types)]
            
            # Generate the problem
            problem_statement, words, probability = generate_probability_problem(
                event_type=current_event_type,
                length=length,
                num_letters=letter_count,
                total_letters=total_letters
            )
            
            # Filter out problems with probability 0 or 1
            numerator, denominator = probability
            if numerator == 0 or numerator == denominator:
                continue
            
            # Extract answer (m + n)
            answer = numerator + denominator
            
            # Create problem entry
            problem_entry = {
                "question": problem_statement,
                "answer": answer,
                "type": "probability",
                "subtype": current_event_type,
                "difficulty": difficulty_level,
                "parameters": {
                    "event_type": current_event_type,
                    "length": length,
                    "total_letters": total_letters,
                    "probability": {"numerator": numerator, "denominator": denominator}
                }
            }
            
            problems.append(problem_entry)
            count += 1
            
            if count % 5 == 0 or count == num_samples:
                print(f"Generated {count}/{num_samples} problems")
            
        except Exception as e:
            continue
    
    if attempts >= max_attempts and count < num_samples:
        print(f"Warning: Could only generate {count}/{num_samples} problems after {max_attempts} attempts")
    
    return problems

def save_problems_to_file(problems, filename, format_type="jsonl"):
    """Save problems to a file.
    
    Args:
        problems: List of problem dictionaries
        filename: Path to the output file
        format_type: Format to save in ("jsonl" or "json")
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if format_type == "jsonl":
        # Write problems to a JSONL file
        with open(filename, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    else:
        # Write problems to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(problems)} problems to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate math problems at a specific difficulty level')
    
    parser.add_argument('--difficulty', type=int, required=True,
                        help='The difficulty level for generated problems (required)')
    parser.add_argument('--type', type=str, choices=['distribution', 'pattern_matching', 'probability', 'all'],
                        default='all', help='Type of problems to generate (default: all)')
    parser.add_argument('--event_type', type=str, 
                        choices=['no_fixed_points', 'no_specific_letter_fixed', 
                                'exactly_n_specific_fixed', 'at_least_n_specific_fixed'],
                        help='Event type for probability problems (only used with --type=probability)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate (default: 10)')
    parser.add_argument('--output_dir', type=str, default="problems/specific_difficulty",
                        help='Output directory for problem files')
    parser.add_argument('--config', type=str, default="difficulty_configs/combi_config.yaml",
                        help='Path to the configuration YAML file')
    parser.add_argument('--format', type=str, choices=['json', 'jsonl'], default='jsonl',
                        help='Output file format (default: jsonl)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate problems based on specified type
    if args.type == 'distribution' or args.type == 'all':
        distribution_problems = generate_distribution_problems(
            args.difficulty, args.num_samples, args.output_dir, config
        )
        save_problems_to_file(
            distribution_problems,
            os.path.join(args.output_dir, f"combinatory_distribution_level_{args.difficulty}.{args.format}"),
            args.format
        )
    
    if args.type == 'pattern_matching' or args.type == 'all':
        pattern_matching_problems = generate_pattern_matching_problems(
            args.difficulty, args.num_samples, args.output_dir, config
        )
        save_problems_to_file(
            pattern_matching_problems,
            os.path.join(args.output_dir, f"combinatory_pattern_matching_level_{args.difficulty}.{args.format}"),
            args.format
        )
    
    if args.type == 'probability' or args.type == 'all':
        # Get event types from config
        event_types = config.get('event_types', [
            'no_fixed_points',
            'no_specific_letter_fixed',
            'exactly_n_specific_fixed',
            'at_least_n_specific_fixed'
        ])
        
        # If specific event type requested, only generate that type
        if args.event_type:
            probability_problems = generate_probability_problems(
                args.difficulty, args.num_samples, args.output_dir, config, args.event_type
            )
            save_problems_to_file(
                probability_problems,
                os.path.join(args.output_dir, f"combinatory_probability_{args.event_type}_level_{args.difficulty}.{args.format}"),
                args.format
            )
        else:
            # Generate each event type as its own category
            all_probability_problems = []
            for event_type in event_types:
                print(f"\nGenerating event type: {event_type}")
                probability_problems = generate_probability_problems(
                    args.difficulty, args.num_samples, args.output_dir, config, event_type
                )
                
                # Save individual event type file
                save_problems_to_file(
                    probability_problems,
                    os.path.join(args.output_dir, f"combinatory_probability_{event_type}_level_{args.difficulty}.{args.format}"),
                    args.format
                )
                
                # Add to combined collection
                all_probability_problems.extend(probability_problems)
            
            # Save a combined file with all probability problems
            save_problems_to_file(
                all_probability_problems,
                os.path.join(args.output_dir, f"combinatory_probability_all_level_{args.difficulty}.{args.format}"),
                args.format
            )
    
    print(f"\nCompleted generating problems at difficulty level {args.difficulty}")

if __name__ == "__main__":
    main() 