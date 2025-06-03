import argparse
import json
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw

from collections import namedtuple
import math

Point = namedtuple('Point',['x','y'])
Segment = namedtuple('Segment',['p','q','dir'])

def perp_intersection(s1: Segment, s2: Segment, eps=1e-6):
    # solve for intersection of the infinite lines through s1 and s2
    # then check the point lies within both segments’ bounding‐boxes
    # return (x,y) or None
    dx1, dy1 = s1.dir
    dx2, dy2 = s2.dir
    denom = dx1*dy2 - dy1*dx2
    if abs(denom) < eps:
        return None      # parallel or nearly so
    # line1: p + t·dir1, line2: r + u·dir2; solve for t,u
    rx, ry = s2.p.x - s1.p.x, s2.p.y - s1.p.y
    t = (rx*dy2 - ry*dx2)/denom
    u = (rx*dy1 - ry*dx1)/denom
    if not (-1e4 <= t <= 1+1e4 and -1e4 <= u <= 1+1e4):
        return None      # intersection lies outside the two segments
    X = s1.p.x + t*dx1
    Y = s1.p.y + t*dy1
    # check perpendicular: dot(dir1, dir2)≈0
    if abs(dx1*dx2 + dy1*dy2) < eps:
        return (X,Y)
    return None

def count_intersections(diagonals, vertices):
    # build Segment objects
    segs = []
    for i,j in diagonals:
        p, q = Point(*vertices[i]), Point(*vertices[j])
        segs.append(Segment(p,q,(q.x-p.x, q.y-p.y)))
    count = 0
    for a in range(len(segs)):
        for b in range(a+1, len(segs)):
            pt = perp_intersection(segs[a], segs[b])
            if pt is not None:
                count += 1
    return count

def create_problem(num_sides=8, num_diagonals=2, line_width=2, seed=0):
    angles = np.linspace(0, 2*np.pi, num_sides + 1)[:-1]
    
    vertices = []
    for angle in angles:
        x = np.cos(angle)
        y = np.sin(angle)
        vertices.append((x, y))
    
    diagonals = []
    for i in range(num_diagonals):
        trials = 0
        while True:
            start_index = np.random.randint(0, num_sides)
            end_index = np.random.randint(0, num_sides)
            if not start_index == end_index and not (start_index + 1) % num_sides == end_index and not (end_index + 1) % num_sides == start_index:
                if not (start_index, end_index) in diagonals and not (end_index, start_index) in diagonals:
                    break
            trials += 1
            if trials > 100:
                return None, None
        diagonals.append((start_index, end_index))
    count = count_intersections(diagonals, vertices)

    problem = f'Suppose you have a {num_sides}-gon, with vertices numbered 1 through {num_sides} in counterclockwise order. '
    
    problem +=  'Draw the diagonal '
    for i in range(num_diagonals):
        problem += f'from vertex {diagonals[i][0] + 1} to vertex {diagonals[i][1] + 1}'
        if i < num_diagonals - 1:
            if num_diagonals != 2:
                problem += ', '
        if i == num_diagonals - 2:
            problem += 'and '
    problem += '.'
    problem += f" How many distinct pairs of diagonals meet at a perpendicular angle (within the polygon's interior or on its boundary)?"
    return problem, count

def generate_dataset(num_images=100, min_diagonals=5, max_diagonals=13, seed=0, output_path=None):
    np.random.seed(seed)
    problems = []
    while len(problems) < num_images:
        while True:
            num_sides = np.random.randint(6, 16)
            # it will be impossible to draw perpendicular diagonals if num_sides is odd
            # but i want a few of these just to have some tricky problems
            if num_sides % 2 == 1:
                if random.random() < 0.95:
                    continue
            break
        num_diagonals = np.random.randint(min_diagonals, max_diagonals + 1)
        newjson = {
            "num_sides": num_sides,
            "num_diagonals": num_diagonals,
            "rotation_vertices": 0,
            "seed": seed
        }
        problem, answer = create_problem(num_sides=num_sides, num_diagonals=num_diagonals, seed=seed)
        if problem is None: # can happen if we try to sample more diagonals than exist
            continue
        newjson["answer_type"] = f"count_perpendicular"
        newjson["question"] = problem
        newjson["answer"] = answer
        problems.append(newjson)
    if output_path is not None:
        with open(output_path, "w") as f:
            for problem in problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    return problems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples per problem type (default: 1000)')
    parser.add_argument('--test_in_samples', type=int, default=200,
                        help='Number of in-distribution test samples per problem type (default: 200)')
    parser.add_argument('--test_out_samples', type=int, default=100,
                        help='Number of out-of-distribution test samples per problem type (default: 100)')
    parser.add_argument('--output_dir', type=str, default="problems",
                        help='Output directory for problem files')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_dataset(num_images=args.train_samples, min_diagonals=7, max_diagonals=13, line_width=3, seed=args.seed, output_path=os.path.join(args.output_dir, "perpendicular_intersection_train.jsonl"))
    generate_dataset(num_images=args.test_in_samples, min_diagonals=7, max_diagonals=13, line_width=3, seed=args.seed, output_path=os.path.join(args.output_dir, "perpendicular_intersection_test_in.jsonl"))
    generate_dataset(num_images=args.test_out_samples, min_diagonals=14, max_diagonals=20, line_width=3, seed=args.seed, output_path=os.path.join(args.output_dir, "perpendicular_intersection_test_out.jsonl"))


if __name__ == "__main__":
    main()

