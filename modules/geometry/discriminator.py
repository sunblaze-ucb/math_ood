import os
import pdb
import sys
import traceback
from typing import Optional
import numpy as np
np.seterr(all='raise') # RuntimeWarnings like divide by zero, degenerate determinants, etc. will now raise exceptions, invalidating some constructions.
import shutil
import time
import argparse
from collections import Counter
from random_constr import Construction
import concurrent.futures
def test_measure_construction(file_path, num_tests=20, precision=4, verbosity=0):
    """
    Test a geometric construction that ends with a measure statement.
    
    Args:
        file_path: Path to the construction file
        num_tests: Number of tests to run
        precision: Number of decimal places to round measurements to
        verbose: Whether to print detailed output
    
    Returns:
        A dictionary with statistics about the measurements
    """
    construction = Construction()
    try:
        construction.load(file_path)
    except Exception as e:
        if verbosity >= 1:
            print(f"Error loading {file_path}: {str(e)}")
            traceback.print_exc()
        return None
    
    if construction.statement_type != "measure":
        if verbosity >= 1:
            print(f"Construction in {file_path} does not end with a measure statement")
        return None
    
    measurements = []
    failures = 0
    
    already_printed = False
    for i in range(num_tests):
        try:
            construction.run_commands()
            value = construction.to_measure.value()
            if value is None:
                print("Got None value from construction: ")
                print(construction.nc_commands)
                print(construction.to_measure)
                continue
            measurements.append(value)
            if verbosity >= 3:
                print(f"Test {i+1}: {value}")
        except Exception as e:
            failures += 1
            if verbosity >= 2 and not already_printed:
                already_printed = True
                print(f"Test {i+1} failed: {str(e)}")
                traceback.print_exc()
    
    if not measurements:
        return None
    
    # Round measurements to specified precision for analysis
    rounded_measurements = [round(m, precision) for m in measurements]
    counts = Counter(rounded_measurements)
    
    # Calculate statistics
    avg = np.mean(measurements)
    median = np.median(measurements)
    min_val = min(measurements)
    max_val = max(measurements)
    
    # Most common value and its count
    most_common = counts.most_common(1)[0]
    mode = most_common[0]
    mode_count = most_common[1]
    
    results = {
        "successful_tests": len(measurements),
        "failed_tests": failures,
        "average": avg,
        "median": median,
        "min": min_val,
        "max": max_val,
        "mode": mode,
        "mode_count": mode_count,
        "all_values": measurements,
        "counts": dict(counts),
        # heuristic: a lot of degenerate constructions are creating measurements that are 0.0
        "pass": mode_count >= 18 and len(measurements) >= 18 and abs(mode) > 0.0001
    }
    
    if verbosity >= 3:
        print(f"\nResults Summary:")
        print(f"Tests: {len(measurements)} successful, {failures} failed")
        print(f"Average: {avg}")
        print(f"Median: {median}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Most common value: {mode} (occurs {mode_count} times out of {len(measurements)})")
        print(f"PASS: {results['pass']}")
    return results

def process_file(directory_path, filename, passed_dir="passed", failed_dir="failed", num_tests=20, verbosity=False, move_files=True):
    file_path = os.path.join(directory_path, filename)
    if verbosity: 
        print(f"Testing {filename}...")
    
    # Test the construction
    test_results = test_measure_construction(file_path, num_tests, verbosity=verbosity)

    if test_results is None:
        if verbosity >= 1: 
            print(f"Failed to test {filename}")
        if move_files:
            shutil.move(file_path, os.path.join(failed_dir, filename))
        return "error", None
    
    passed = test_results["pass"]
    
    # Move file to appropriate directory
    destination = os.path.join(passed_dir if passed else failed_dir, filename)
    if move_files:
        shutil.move(file_path, destination)
    
    if verbosity >= 1: 
        print(f"{'PASSED' if passed else 'FAILED'}: {test_results['mode_count']} of {test_results['successful_tests']} tests gave the same result")
    
    if passed:
        return "pass", test_results["mode"]
    else:
        return "fail", None


def parse_args():
    parser = argparse.ArgumentParser(description="Test geometric constructions")
    parser.add_argument("--path", default="generated_constructions/", help="Path to construction file or directory")
    parser.add_argument("--verbosity", type=int, default=0, help="Print verbose output")
    parser.add_argument("--num_tests", type=int, default=20, help="Number of tests to run")
    parser.add_argument("--nomovefiles", action="store_false", dest="move_files", help="Don't move files to passed/ or failed/")
    parser.add_argument("--max_workers", type=int, default=16, help="Maximum number of processes to use")
    parser.add_argument("--multiprocess", action="store_true", help="Run tests in parallel")
    args = parser.parse_args()
    return args

def main(args):
    
    if os.path.isdir(args.path):
        timestamp = int(time.time())
        print(f"Output timestamp: {timestamp}")
        passed_dir = os.path.join("passed/", str(timestamp))
        failed_dir = os.path.join("failed/", str(timestamp))

        if args.move_files:
            os.makedirs(passed_dir, exist_ok=True)
            os.makedirs(failed_dir, exist_ok=True)
            
        
        all_answers = []
        
        num_passed = 0
        num_failed = 0
        num_errors = 0
        if not args.multiprocess:
            # Run sequentially
            for filename in os.listdir(args.path):
                if not filename.endswith('.txt'):
                    continue
                results = process_file(args.path, filename, passed_dir=passed_dir, failed_dir=failed_dir, num_tests=args.num_tests, verbosity=args.verbosity, move_files=args.move_files)
                if results[0] == "pass":
                    all_answers.append(f"{filename}: {results[1]}")
                    num_passed += 1
                elif results[0] == "fail":
                    num_failed += 1
                else:
                    num_errors += 1
        else:
            # Use ProcessPool for parallel processing
            files_to_process = [f for f in os.listdir(args.path) if f.endswith('.txt')]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        process_file, 
                        args.path, 
                        filename, 
                        passed_dir=passed_dir, 
                        failed_dir=failed_dir, 
                        num_tests=args.num_tests, 
                        verbosity=args.verbosity, 
                        move_files=args.move_files
                    ): filename for filename in files_to_process
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        results = future.result()
                        if results[0] == "pass":
                            all_answers.append(f"{filename}: {results[1]}")
                            num_passed += 1
                        elif results[0] == "fail":
                            num_failed += 1
                        else:
                            num_errors += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        num_errors += 1

        if args.move_files:
            answers_path = os.path.join(passed_dir, "answers.txt")
            with open(answers_path, 'a') as f:
                for answer in all_answers:
                    f.write(f"{answer}\n")
        print(f"\nSummary: {num_passed} passed, {num_failed} failed, {num_errors} errors")
    else:
        if args.verbosity < 2:
            args.verbosity = 2
        test_measure_construction(args.path, args.num_tests, verbosity=args.verbosity) 
    return timestamp

if __name__ == "__main__":
    args = parse_args()
    main(args)