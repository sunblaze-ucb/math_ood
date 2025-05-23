
# count shortest path from bottom left to top right
# while only moving using a knight's move
import argparse
from typing import Tuple, List
import os
import json
import random

def generate_generalized_knight_move_problem(width, height, move_x, move_y, target_x, target_y):
    """
    Generate a random knight move problem
    """
    # Initialize the grid
    # Define the knight's possible moves
    moves = [
        (-move_x, -move_y), (-move_x, move_y), (-move_y, -move_x), (-move_y, move_x),
        (move_x, -move_y), (move_x, move_y), (move_y, -move_x), (move_y, move_x)
    ]
    
    start = (0, 0)
    target = (target_x, target_y)
    
    # Use BFS to find shortest path
    queue = [(start, 0)]  # (position, distance)
    visited = set([start])
    
    answer = None
    while queue:
        (x, y), distance = queue.pop(0)
        
        # If we reached the target, return the distance
        if (x, y) == target:
            answer = distance
            break
        
        # Try all possible knight moves
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within bounds and not visited
            if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    if answer is None:
        return None
    
    target_str = f"({target_x}, {target_y})"
    if target_x == width - 1 and target_y == height - 1:
        target_str = "the top right"
    if move_x == 1 and move_y == 2 or move_x == 2 and move_y == 1:
        question = f"Supposing you start in the bottom left (0, 0) cell of a {width}x{height} grid, what is minimum number of steps you can take to get to {target_str}, if your only valid moves are like a chess knight, i.e, over two squares in one direction and over one square in a perpendicular direction in one move?"
    else:
        question = f"Supposing you start in the bottom left (0, 0) cell of a {width}x{height} grid, what is minimum number of steps you can take to get to {target_str}, if your only valid moves are to move over {move_x} squares in one direction and over {move_y} squares in a perpendicular direction (at once)? (Assume you cannot move off the grid.)"

    json = {
        "question": question,
        "answer": answer,
        "width": width,
        "height": height,
        "type": "knight_move"
    }
    return json

def generate_knight_problems(args):
    problems_done = {} # note this is outside the function; we don't want to repeat between train/val
    # for the other functions in this file the probability is negligible but there aren't that many valid problems for this one.
    def generate_n_problems(min_dim, max_dim, n):
        strings = []
        # knight-ish problems
        while len(strings) < n:
            width = random.randint(min_dim, max_dim)
            height = random.randint(min_dim, max_dim)
            move_x = random.randint(1, 6)
            move_y = random.randint(1, 6)
            if move_x == move_y:
                continue
            if (move_x, move_y) in ((2, 4), (4, 2)):
                continue
            if move_x + move_y > 7:
                continue
            while True:
                target_x = random.randint(0, width)
                target_y = random.randint(0, height)
                if target_x + target_y > width + height - 8:
                    break
            if (width, height, move_x, move_y, target_x, target_y) in problems_done:
                continue
            question_info = generate_generalized_knight_move_problem(width, height, move_x, move_y, target_x, target_y)
            if question_info is None:
                continue
            if question_info["answer"] < 4:
                continue
            strings.append(json.dumps(question_info))
            problems_done[(width, height, move_x, move_y, target_x, target_y)] = True
        return strings
    with open(os.path.join(args.output_dir, f"logic_gridworld_knight_move_7B_train.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(3, 9, args.train_samples)))
    with open(os.path.join(args.output_dir, f"logic_gridworld_knight_move_7B_test_in.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(3, 9, args.test_in_samples)))
    with open(os.path.join(args.output_dir, f"logic_gridworld_knight_move_7B_test_out.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(9, 12, args.test_out_samples)))


def generate_blocked_gridworld_problem(width, height, blocked_cells: List[Tuple[int, int]]):
    """
    Calculate the number of paths from the bottom left (0, 0) to the top right (width-1, height-1)
    of a grid, avoiding any blocked cells.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        blocked_cells: List of (x, y) coordinates of blocked cells
    
    Returns:
        A dictionary containing the problem information
    """
    # Create a grid to store the number of paths to each cell
    dp = [[0 for _ in range(width)] for _ in range(height)]
    
    # Convert blocked cells to a set for O(1) lookup
    blocked = set(blocked_cells)
    
    # Base case: there is 1 way to reach the starting position (0, 0)
    # unless it's blocked
    if (0, 0) not in blocked:
        dp[0][0] = 1
    
    # Fill the dp table
    for y in range(height):
        for x in range(width):
            # Skip if current cell is blocked or it's the starting cell (already handled)
            if (x, y) in blocked or (x == 0 and y == 0):
                continue
            
            # We can reach the current cell from the left or from below
            if x > 0 and (x-1, y) not in blocked:
                dp[y][x] += dp[y][x-1]
            if y > 0 and (x, y-1) not in blocked:
                dp[y][x] += dp[y-1][x]
    
    # The answer is the number of paths to the top right
    answer = dp[height-1][width-1]
    
    # Create the question string
    blocked_str = ", ".join([f"({x}, {y})" for x, y in blocked_cells])
    question = f"In a {width}x{height} grid, how many different paths are there from the bottom left (0, 0) to the top right ({width-1}, {height-1}), if you can only move right or up at each step, subject to the constraint that you cannot move through the following cells: {blocked_str}?"
    
    return {
        "question": question,
        "answer": answer,
        "width": width,
        "height": height,
        "num_blocked_cells": len(blocked_cells),
        "type": "blocked_gridworld"
    }

    
def generate_blocked_gridworld_problems(args):
    def generate_n_problems(min_dim, max_dim, n, blocked_cell_multiplier=1.2):
        strings = []
        while len(strings) < n:
            width = random.randint(min_dim, max_dim)
            height = random.randint(min_dim, max_dim)
            if width + height < 8:
                continue
            num_blocked_cells = random.randint(3, int(width * blocked_cell_multiplier))
            blocked_cells = []
            for i in range(num_blocked_cells):
                cell = (random.randint(0, width - 1), random.randint(0, height - 1))
                while True:
                    cell = (random.randint(0, width - 1), random.randint(0, height - 1))
                    if cell not in blocked_cells and cell != (0, 0) and cell != (width - 1, height - 1):
                        blocked_cells.append(cell)
                        break
            strings.append(json.dumps(generate_blocked_gridworld_problem(width, height, blocked_cells)))
        return strings
    with open(os.path.join(args.output_dir, f"logic_gridworld_blocked_7B_train.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(3, 8, args.train_samples)))
    with open(os.path.join(args.output_dir, f"logic_gridworld_blocked_7B_test_in.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(3, 8, args.test_in_samples)))
    with open(os.path.join(args.output_dir, f"logic_gridworld_blocked_7B_test_out.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(8, 10, args.test_out_samples, 1.5)))

def generate_rookmove_gridworld_problem(width, height, blocked_cells: List[Tuple[int, int]]):
    """
    Calculate the number of paths from the bottom left (0, 0) to the top right (width-1, height-1)
    of a grid, avoiding any blocked cells.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        blocked_cells: List of (x, y) coordinates of blocked cells
    
    Returns:
        A dictionary containing the problem information
    """
    # Create a grid to store the number of paths to each cell
    dp = [[0 for _ in range(width)] for _ in range(height)]
    
    # Convert blocked cells to a set for O(1) lookup
    blocked = set(blocked_cells)
    
    # Initialize dp with infinity for all cells
    dp = [[float('inf') for _ in range(width)] for _ in range(height)]
    
    # Base case: starting position (0, 0) takes 0 steps to reach itself
    if (0, 0) not in blocked:
        dp[0][0] = 0
    
    # Fill the dp table
    for y in range(height):
        for x in range(width):
            # Skip if current cell is blocked
            if (x, y) in blocked:
                continue
            
            # Try all possible rook moves to reach current cell
            # Move horizontally (from left)
            for prev_x in range(x):
                # Check if path is clear (no blocked cells in between)
                if all((i, y) not in blocked for i in range(prev_x + 1, x)):
                    dp[y][x] = min(dp[y][x], dp[y][prev_x] + 1)
            
            # Move vertically (from below)
            for prev_y in range(y):
                # Check if path is clear (no blocked cells in between)
                if all((x, j) not in blocked for j in range(prev_y + 1, y)):
                    dp[y][x] = min(dp[y][x], dp[prev_y][x] + 1)
    
    # The answer is the minimum steps to the top right
    # If unreachable, answer will be infinity
    answer = dp[height-1][width-1] if dp[height-1][width-1] != float('inf') else -1
    
    # Create the question string
    blocked_str = ", ".join([f"({x}, {y})" for x, y in blocked_cells])
    question = f"In a {width}x{height} grid, what is the minimum number of steps it takes to move from the bottom left (0, 0) to the top right ({width-1}, {height-1}), provided you can move right or up by any number of cells (like a chess rook) at each step, subject to the constraint that you cannot move through the following cells: {blocked_str}? If it's impossible, answer with -1."
    
    return {
        "question": question,
        "answer": answer,
        "width": width,
        "height": height,
        "num_blocked_cells": len(blocked_cells),
        "type": "rookmove_gridworld"
    }

def generate_rookmove_gridworld_problems(args):
    def generate_n_problems(min_dim, max_dim, n, min_blocked_proportion=0.1, max_blocked_proportion=0.2):
        strings = []
        for _ in range(n):
            width = random.randint(min_dim, max_dim)
            height = random.randint(min_dim, max_dim)
            num_blocked_cells = random.randint(int(width * height * min_blocked_proportion), int(width * height * max_blocked_proportion))
            if num_blocked_cells < min(width, height):
                num_blocked_cells = min(width, height)
            blocked_cells = []
            for i in range(num_blocked_cells):
                cell = (random.randint(0, width - 1), random.randint(0, height - 1))
                while True:
                    cell = (random.randint(0, width - 1), random.randint(0, height - 1))
                    if cell not in blocked_cells and cell != (0, 0) and cell != (width - 1, height - 1):
                        blocked_cells.append(cell)
                        break
                if not any(i == 0 for (i, j) in blocked_cells):
                    blocked_cells.append((random.randint(1, width - 1), 0))
                if not any(j == 0 for (i, j) in blocked_cells):
                    blocked_cells.append((0, random.randint(1, height - 1)))
            strings.append(json.dumps(generate_rookmove_gridworld_problem(width, height, blocked_cells)))
        return strings
    with open(os.path.join(args.output_dir, f"logic_gridworld_rookmove_7B_train.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(3, 6, args.train_samples)))
    with open(os.path.join(args.output_dir, f"logic_gridworld_rookmove_7B_test_in.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(3, 6, args.test_in_samples)))
    with open(os.path.join(args.output_dir, f"logic_gridworld_rookmove_7B_test_out.jsonl"), 'w') as f:
        f.write("\n".join(generate_n_problems(6, 9, args.test_out_samples, 0.1, 0.3)))

def main():
    parser = argparse.ArgumentParser(description='Generate function analysis problems')

    # Add arguments for sample counts
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples per problem type (default: 1000)')
    parser.add_argument('--test_in_samples', type=int, default=200,
                        help='Number of in-distribution test samples per problem type (default: 200)')
    parser.add_argument('--test_out_samples', type=int, default=100,
                        help='Number of out-of-distribution test samples per problem type (default: 100)')
    parser.add_argument('--output_dir', type=str, default="problems",
                        help='Output directory for problem files')

    args = parser.parse_args()

    generate_knight_problems(args)
    generate_blocked_gridworld_problems(args)
    generate_rookmove_gridworld_problems(args)

if __name__ == "__main__":
    main()