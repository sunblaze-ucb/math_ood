import argparse
from typing import Tuple, List
import os
import json
import random


def generate_max_num_color_cells_problem(size, num_constraints: int=None):
    color = random.choice(["black", "white"])
    other_color = "white" if color == "black" else "black"
    grid = [[None for _ in range(size)] for _ in range(size)]
    if num_constraints is None:
        num_constraints = random.randint(size - 2, size + 3)
    constraints = []
    for _ in range(num_constraints):
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        grid[row][col] = other_color
        constraints.append((row, col))
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            matched_constraint = False
            for constraint in constraints:
                if constraint[0] == row_idx or constraint[1] == col_idx:
                    matched_constraint = True
                    break
            if matched_constraint:
                continue
            grid[row_idx][col_idx] = color
    answer = sum(sum(1 for cell in row if cell is not None) for row in grid)

    while True:
        to_remove = None
        
        for idx, cell in enumerate(constraints):
            # if cell shares a row with a constraint of same color and col of constraint with same color, it's redundant
            match_row = False
            match_col = False
            for constraint in constraints:
                if constraint == cell:
                    continue
                if constraint[0] == cell[0]:
                    match_row = True
                if constraint[1] == cell[1]:
                    match_col = True
            if match_row and match_col:
                to_remove = cell
                break
        if to_remove is not None:
            constraints.remove(to_remove)
        else:
            break
    constraint_str = ""
    for constraint in constraints:
        constraint_str += f"cell ({constraint[0]}, {constraint[1]}) is {other_color}, "
    constraint_str = constraint_str[:-2] # remove trailing comma and space
    question = f"Chips, colored either black or white, are placed in a {size}x{size} grid such that: a) each cell contains at most one chip, b) all chips in the same row and all chips in the same column have the same colour. Furthermore, we have the following constraints (with the cells 0-indexed): {constraint_str}. What is the maximum number of {color} chips that can be placed on the grid?"

    return {
        "question": question,
        "answer": answer,
        "type": "max_num_color_cells"
    }
    
def generate_count_fully_constrained_problem(grid_size):
    while True:
        row_colors = [random.choice(["black", "white"]) for _ in range(grid_size)]
        col_colors = [random.choice(["black", "white"]) for _ in range(grid_size)]
        if all(x == row_colors[0] for x in row_colors) or all(x == col_colors[0] for x in col_colors):
            continue
        break

    grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    for i in range(grid_size):
        for j in range(grid_size):
            if row_colors[i] == col_colors[j]:
                grid[i][j] = row_colors[i]

    answer = sum(sum(1 for cell in row if cell is not None) for row in grid)
    constraints = []
    while True:
        cell = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if grid[cell[0]][cell[1]] is None:
            continue
        if cell in constraints:
            continue
        constraints.append(cell)
        row_covered = [False for _ in range(grid_size)]
        col_covered = [False for _ in range(grid_size)]
        for constraint in constraints:
            row_covered[constraint[0]] = True
            col_covered[constraint[1]] = True
        if all(row_covered) and all(col_covered):
            break
    while True:
        to_remove = None
        
        for idx, cell in enumerate(constraints):
            # if cell shares a row with a constraint of same color and col of constraint with same color, it's redundant
            match_row_of_same_color = False
            match_col_of_same_color = False
            color_at_cell = grid[cell[0]][cell[1]]
            for constraint in constraints:
                if constraint == cell:
                    continue
                color_at_constraint = grid[constraint[0]][constraint[1]]
                if color_at_constraint == color_at_cell:
                    if constraint[0] == cell[0]:
                        match_row_of_same_color = True
                    if constraint[1] == cell[1]:
                        match_col_of_same_color = True
            if match_row_of_same_color and match_col_of_same_color:
                to_remove = cell
                break
        if to_remove is not None:
            constraints.remove(to_remove)
        else:
            break
        
    # Create the question string
    constraint_str = ""
    for constraint in constraints:
        constraint_str += f"cell ({constraint[0]}, {constraint[1]}) is {grid[constraint[0]][constraint[1]]}, "
    constraint_str = constraint_str[:-2] # remove trailing comma and space
    question = f"Chips, colored either black or white, are placed in the cells of a {grid_size}x{grid_size} grid such that: a) each cell contains at most one chip, b) all chips in the same row and all chips in the same column have the same colour, c) any additional chip placed on the grid would violate one or more of the previous two conditions. Furthermore, we have the following constraints (with the cells 0-indexed): {constraint_str}. How many chips are placed on the grid?"
    # question = f"There is a collection of 25 indistinguishable white chips and 25 indistinguishable black chips. Find the number of ways to place some of these chips in the 25 unit cells of a 5x5 grid such that: each cell contains at most one chip all chips in the same row and all chips in the same column have the same colour any additional chip placed on the grid would violate one or more of the previous two conditions."
    
    return {
        "question": question,
        "answer": answer,
        "type": "count_fully_constrained"
    }


def generate_n_constraint_problems(n, grid_size):
    strings = []
    while len(strings) < n:
        if random.random() < 0.5:
            problem = generate_max_num_color_cells_problem(grid_size)
        else:
            problem = generate_count_fully_constrained_problem(grid_size)
        strings.append(json.dumps(problem))
    return strings

# for fine-tuning for the compositional problem
def generate_particular_ft_dataset(n):
    strings = []
    while len(strings) < n:
        if random.random() < 0.5:
            problem = generate_max_num_color_cells_problem(5)
        else:
            problem = generate_count_fully_constrained_problem(5)
        strings.append(json.dumps(problem))
    return strings

def generate_constraint_maximization_problems(args):
    with open(os.path.join(args.output_dir, f"constraint_maximization_train.jsonl"), 'w') as f:
        f.write("\n".join(generate_particular_ft_dataset(args.train_samples)))

def main():
    parser = argparse.ArgumentParser(description='Generate function analysis problems')

    # Add arguments for sample counts
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples per problem type (default: 1000)')
    parser.add_argument('--test_in_samples', type=int, default=100,
                        help='Number of in-distribution test samples per problem type (default: 200)')
    parser.add_argument('--test_out_samples', type=int, default=50,
                        help='Number of out-of-distribution test samples per problem type (default: 100)')
    parser.add_argument('--output_dir', type=str, default="difficulty",
                        help='Output directory for problem files')
    args = parser.parse_args()

    generate_constraint_maximization_problems(args)
if __name__ == "__main__":
    main()