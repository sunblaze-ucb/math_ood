import math
import os
import argparse
import glob
from typing import List, Optional, Dict
from pathlib import Path
import pdb
import time
import hashlib
import time
import json
import concurrent.futures
import threading
from base_translate import translate_problem as base_translate_problem
from missing_angle_translate import translate_problem as missing_angle_translate_problem
global_timestamp = str(int(time.time()))


def process_file_contents(filename: str, contents: str, answer: str, output_dir: Path, hash: str, translator_type: str = "base") -> None:
    if translator_type == "base":
        problem_lines, stats = base_translate_problem(contents)
    elif translator_type == "missing_angle":
        problem_lines, stats, answer = missing_angle_translate_problem(contents, answer=answer)
    else:
        raise ValueError(f"Invalid translator type: {translator_type}")
    if problem_lines is None:
        return
    problem = " ".join(problem_lines)
    return json.dumps({"question": problem, "answer": answer, "hash": hash, "original_filename": filename, "stats": stats})


def process_timestamp_dirs(args: argparse.Namespace) -> None:
    """
    Search the 'passed' directory for subdirectories that look like timestamps
    and process their contents if they come after the specified timestamp.
    
    Args:
        after: Optional timestamp to filter directories (process only dirs with 
               timestamps greater than this value)
        hashes: Dictionary of file hashes that have already been processed
        max_workers: Maximum number of parallel workers to use
    """
    os.makedirs(args.output_dir, exist_ok=True)
    if args.hash_check:
        hashes = read_hashes(args.output_dir)
    else:
        hashes = {}
    passed_dir = "passed"
    
    # Get all subdirectories that look like timestamps
    if args.interpret_timestamp_as_after:
        timestamp_dirs = []
        for item in os.listdir(passed_dir):
            item_path = os.path.join(passed_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                file_ts = int(item)
                if args.timestamp is None or file_ts >= args.timestamp:
                    timestamp_dirs.append((file_ts, item_path))
    else:
        timestamp_dirs = [(args.timestamp, os.path.join(passed_dir, str(args.timestamp)))]

        
    # Sort directories by timestamp
    timestamp_dirs.sort()
    
    # Create a list to hold all tasks
    all_tasks = []
    
    # Process each directory
    for timestamp, dir_path in timestamp_dirs:
        print(f"Processing directory: {dir_path} (timestamp: {timestamp})")
        # Process all files in the directory
        with open(os.path.join(dir_path, "answers.txt"), 'r') as f:
            answer_lines = f.read().strip().split("\n")
        for file_path in glob.glob(os.path.join(dir_path, "*.txt")):
            if "answers.txt" in file_path:
                continue
            filename = os.path.basename(file_path)
            contents = open(file_path, 'r').read()
            hash = hashlib.sha256(contents.encode()).hexdigest()
            if args.hash_check and hash in hashes:
                continue
            for line in answer_lines:
                if line.startswith(filename):
                    answer = line.split(" ")[1]
                    break
            all_tasks.append((f"{dir_path}/{filename}", contents, answer, args.output_dir, hash, args.translator_type))
    
    if args.translator_type == "base":
        name_extension = "mechanically_translated"
    elif args.translator_type == "missing_angle":
        name_extension = "missing_angle"
    else:
        raise ValueError(f"Invalid translator type: {args.translator_type}")

    all_problem_json_strings = []
    # Process all tasks in parallel
    if args.sequential:
        for task in all_tasks:
            problem_json_string = process_file_contents(*task)
            all_problem_json_strings.append(problem_json_string)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_file_contents, *task): task[0] for task in all_tasks}
            for future in concurrent.futures.as_completed(futures):
                problem_json_string = future.result()
                if problem_json_string is None:
                    continue
                all_problem_json_strings.append(problem_json_string)

    output_file = os.path.join(args.output_dir, f"{global_timestamp}_{name_extension}.jsonl")
    if args.output_name:
        output_file = os.path.join(args.output_dir, f"{args.output_name}")
    count = 0
    if args.return_problems:
        return all_problem_json_strings
    with open(output_file, 'a') as f:
        for problem_json_string in all_problem_json_strings:
            if problem_json_string is None:
                continue
            f.write(problem_json_string + "\n")
            count += 1

    print (f"Wrote {count} translations to: {output_file}")
    return output_file


def read_hashes(output_dir: Path) -> Dict[str, bool]:
    hashes = {}
    for file in os.listdir(output_dir):
        # bit hacky-- heuristic to ignore files that are filtered, e.g, _validated.jsonl files
        if file.endswith(".jsonl") and not "_" in file:
            with open(os.path.join(output_dir, file), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    hashes[data["hash"]] = True
    return hashes

def parse_args():
    parser = argparse.ArgumentParser(description="Process timestamp directories in the 'passed' folder.")
    parser.add_argument("--interpret_timestamp_as_after", action="store_true",
                        help="Interpret timestamp as after this value")
    parser.add_argument("--timestamp", type=int, default=None,
                        help="Only process the given timestamp")
    parser.add_argument("--output_dir", type=Path, default=Path("natural_language_problems"), 
                        help="Place to dump translated files.")
    parser.add_argument("--output_name", type=str, default=None, 
                        help="Name of the output file.")
    parser.add_argument("--nohashcheck", action="store_false", dest="hash_check",
                        help="Disable hash checking")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of parallel workers")
    parser.add_argument("--sequential", action="store_true",
                        help="Process files singlethreaded (for debugging)")
    parser.add_argument("--translator_type", type=str, choices=["missing_angle", "base"], default="base",
                        help="Type of translator to use")
    parser.add_argument("--return_problems", type=bool, default=False,
                        help="Return problems instead of writing to file")
    args = parser.parse_args()
    return args

def main(args) -> None:
    output_file = process_timestamp_dirs(args)
    return output_file


if __name__ == "__main__":
    args = parse_args()
    main(args)