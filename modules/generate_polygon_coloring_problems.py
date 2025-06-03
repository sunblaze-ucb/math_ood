
# count shortest path from bottom left to top right
# while only moving using a knight's move
import argparse
from typing import Tuple, List
import os
import json
import random


def generate_num_colorings_problem(min_vertices, max_vertices):
    num_vertices = random.randint(min_vertices, max_vertices)
    vertex_colors = [random.choice(["red", "blue"]) for vertex in range(num_vertices)]
    # Select three random indices to set to None
    indices_to_remove = random.sample(range(num_vertices), 3)
    for idx in indices_to_remove:
        vertex_colors[idx] = "uncolored"
    color = random.choice(["red", "blue"])
    other_color = "red" if color == "blue" else "blue"

    def exists_rotation_covering(coloring):
        for i in range(num_vertices):
            is_valid_coloring = True
            for j in range(num_vertices):
                # rotate by i, check to see if all {color} vertices end up at positions where there were originally {other_color} vertices
                if coloring[(j + i) % num_vertices] == color and coloring[j] == color:
                    is_valid_coloring = False
                    break
            if is_valid_coloring:
                return True
        return False
    # Generate all possible colorings for the 3 removed indices
    count = 0
    for color1 in ["red", "blue"]:
        for color2 in ["red", "blue"]:
            for color3 in ["red", "blue"]:
                current_coloring = vertex_colors.copy()
                current_coloring[indices_to_remove[0]] = color1
                current_coloring[indices_to_remove[1]] = color2
                current_coloring[indices_to_remove[2]] = color3
                if exists_rotation_covering(current_coloring):
                    count += 1
    answer = count

    vertex_colors_str = ", ".join([f"vertex {i} is {vertex_colors[i]}" for i in range(num_vertices)])
    question = f"A {num_vertices}-gon is colored so that in clockwise order, the vertices are colored as follows: {vertex_colors_str}. How many valid colorings of the uncolored vertices exist such that the {num_vertices}-gon can then be rotated in such a way that all of the {color} vertices end up at positions where there were originally {other_color} vertices?"

    return {
        "question": question,
        "answer": answer,
        "type": "max_num_color_cells"
    }
    
def generate_num_covered_problem(min_vertices, max_vertices):
    num_vertices = random.randint(min_vertices, max_vertices)
    vertex_colors = [random.choice(["red", "blue"]) for vertex in range(num_vertices)]
    color = random.choice(["red", "blue"])
    other_color = "red" if color == "blue" else "blue"

    num_rotations = random.randint(1, num_vertices - 1)
    rotated_vertex_colors = [vertex_colors[(i + num_rotations) % num_vertices] for i in range(num_vertices)]
    count = 0
    for i in range(num_vertices):
        if rotated_vertex_colors[i] == color and vertex_colors[i] == other_color:
            count += 1
    answer = count

    vertex_colors_str = ", ".join([f"vertex {i} is {vertex_colors[i]}" for i in range(num_vertices)])
    question = f"A {num_vertices}-gon is colored so that in clockwise order, the vertices are colored as follows: {vertex_colors_str}. The {num_vertices}-gon is then rotated clockwise by {num_rotations} positions, so that the vertex that was originally in position 0 is now in position {num_rotations}. How many {color} vertices are now in positions where there were originally {other_color} vertices?"
    
    return {
        "question": question,
        "answer": answer,
        "type": "count_fully_constrained"
    }

def generate_max_cover_problem(min_vertices, max_vertices):
    num_vertices = random.randint(min_vertices, max_vertices)
    vertex_colors = [random.choice(["red", "blue"]) for vertex in range(num_vertices)]
    color = random.choice(["red", "blue"])
    other_color = "red" if color == "blue" else "blue"

    max_count = 0
    for num_rotations in range(1, num_vertices):
        rotated_vertex_colors = [vertex_colors[(i + num_rotations) % num_vertices] for i in range(num_vertices)]
        count = 0
        for i in range(num_vertices):
            if rotated_vertex_colors[i] == color and vertex_colors[i] == other_color:
                count += 1
        if count > max_count:
            max_count = count
    answer = max_count

    vertex_colors_str = ", ".join([f"vertex {i} is {vertex_colors[i]}" for i in range(num_vertices)])
    question = f"A {num_vertices}-gon is colored so that in clockwise order, the vertices are colored as follows: {vertex_colors_str}. What is the maximum number of {color} vertices that can be made to occupy a position where there were originally {other_color} vertices by rotating the {num_vertices}-gon?"
    
    return {
        "question": question,
        "answer": answer,
        "type": "count_fully_constrained"
    }

def generate_n_constraint_problems(n, min_vertices, max_vertices):
    problems = []
    while len(problems) < n:
        random_number = random.random()
        if random_number < 0.4:
            problem = generate_num_colorings_problem(min_vertices, max_vertices)
        elif random_number < 0.6:
            problem = generate_num_covered_problem(min_vertices, max_vertices)
        else:
            problem = generate_max_cover_problem(min_vertices, max_vertices)
        problems.append(problem)
    return problems

def generate_constraint_maximization_problems(args):
    with open(os.path.join(args.output_dir, f"polygon_coloring_train.jsonl"), 'w') as f:
        for problem in generate_n_constraint_problems(args.train_samples, args.min_vertices, args.max_vertices):
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')

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