import argparse
import json
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import pdb

def create_polygon_image(size=500, save_path="polygon_diagrams", idx=0, num_sides=8, num_diagonals=2, rotation_vertices=1, line_width=2, seed=0):
    os.makedirs(save_path, exist_ok=True)
    # Calculate vertices
    center = size // 2
    radius = size * 0.4  # 40% of image size
    angles = np.linspace(0, 2*np.pi, num_sides + 1)[:-1]
    
    vertices = []
    for angle in angles:
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        vertices.append((x, y))
    
    # Create a new PIL image
    img_pil = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img_pil)
    
    draw.polygon(vertices, outline='black', fill=None, width=line_width)
    

    # Draw the diagonals
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
                return None
        diagonals.append((start_index, end_index))
    
    for start, end in diagonals:
        draw.line([vertices[start], vertices[end]], fill='black', width=line_width)

    rotated_diagonals = []
    for start, end in diagonals:
        rotated_diagonals.append(((start + rotation_vertices) % num_sides, (end + rotation_vertices) % num_sides))

    for start, end in rotated_diagonals:
        draw.line([vertices[start], vertices[end]], fill='red', width=line_width)

    save_path = f"{save_path}/polygon_{idx}.png"
    img_pil.save(save_path)

    problem = f'Suppose you have a {num_sides}-gon, with vertices numbered 1 through {num_sides} in counterclockwise order. '
    
    problem +=  'Draw the diagonal'
    for i in range(num_diagonals):
        problem += f' from vertex {diagonals[i][0] + 1} to vertex {diagonals[i][1] + 1}'
        if i < num_diagonals - 1:
            if num_diagonals != 2:
                problem += ','
        if i == num_diagonals - 2:
            problem += ' and '
    problem += '.'

    problem +=  f' Then, rotate the entire setup, including the constructed diagonals, {rotation_vertices} vertices counterclockwise (so that vertex 1 ends up where vertex {rotation_vertices + 1} was), and superimpose it on the original (so that the resulting diagram contains both the original diagonals and the rotated versions of the diagonals). The original {num_sides}-gon will be partitioned into a collection of smaller polygons. (Only count the "smallest" polygons, i.e, those that do not have any of the diagonals running through them; this is not a combinatorics problem.)'
    return save_path, problem


def count_connected_components(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

    real_labels = []
    for idx, component_size in enumerate(stats[:, 4]):
        if component_size > 10:
            real_labels.append(idx)

    approxs = []
    for component_label in real_labels[2:]:
        # Create binary mask using numeric comparison, not string comparison
        binary_mask = (labels == component_label).astype(np.uint8)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours were found
        if len(contours) > 0:
            # Method 1: Use approxPolyDP to approximate the contour to the hexagon corners
            epsilon = 0.005 * cv2.arcLength(contours[0], True)  # Precision parameter
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            approxs.append(len(approx))
        else:
            print(f"No contours found for label {component_label}")
    '''
    # for debugging-- look directly at the labels in an ascii art image
    labels = labels.astype(str)
    for i, row in enumerate(labels):
        for j, entry in enumerate(row):
            for k in range(26):
                if entry == str(k + 10):
                    labels[i][j] = chr(k + 65)

    np.savetxt(f'{image_path}_labels.txt', labels, fmt='%s')
    return len(real_labels)
    '''
    return approxs

def generate_dataset(num_images=100, min_diagonals=1, max_diagonals=4, min_sides=6, max_sides=13, seed=0, output_path=None):
    np.random.seed(seed)
    problems = []
    for i in range(num_images):
        num_sides = np.random.randint(min_sides, max_sides)
        num_diagonals = np.random.randint(min_diagonals, max_diagonals + 1)
        if num_diagonals == 1 and random.random() < 0.75: # just a manual hack. there were too many of these and they are too easy.
            num_diagonals = 2
        rotation_vertices = np.random.randint(1, num_sides)
        newjson = {
            "num_sides": num_sides,
            "num_diagonals": num_diagonals,
            "rotation_vertices": rotation_vertices,
            "seed": seed
        }
        image_path, problem = create_polygon_image(size=10000, idx=i, num_sides=num_sides, num_diagonals=num_diagonals, rotation_vertices=rotation_vertices, line_width=3, seed=seed)
        if image_path is None:
            continue
        polygon_sides = count_connected_components(image_path)

        a = random.random()
        if a < 0.0:
            problem += " How many such polygons will there be?"
            newjson["answer_type"] = "count"
            answer = len(polygon_sides)
        else:
            options = list(set(polygon_sides))
            random.shuffle(options)
            choice = options[0]
            problem += f" How many such polygons will there be with {choice} sides?"
            newjson["answer_type"] = f"count_{choice}_sides"
            answer = polygon_sides.count(choice)
        newjson["question"] = problem
        newjson["answer"] = answer
        problems.append(newjson)
    if output_path is not None:
        with open(output_path, "w") as f:
            for problem in problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    return problems

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
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    generate_dataset(num_images=args.train_samples, min_diagonals=1, max_diagonals=3, seed=args.seed, output_path=os.path.join(args.output_dir, "polygon_rotation_train.jsonl"))
    generate_dataset(num_images=args.test_in_samples, min_diagonals=1, max_diagonals=3, seed=args.seed, output_path=os.path.join(args.output_dir, "polygon_rotation_test_in.jsonl"))
    generate_dataset(num_images=args.test_out_samples, min_diagonals=4, max_diagonals=4, seed=args.seed, output_path=os.path.join(args.output_dir, "polygon_rotation_test_out.jsonl"))


# debug
def draw_polygon(num_sides=8, line_width=2):
    angles = np.linspace(0, 2*np.pi, num_sides + 1)[:-1]
    vertices = [(np.cos(angle), np.sin(angle)) for angle in angles]
    
    diagonals = [(6, 4), (1, 7), (5, 0), (4, 1), (4, 0), (2, 5), (6, 3), (5, 3), (1, 3), (5, 7), (5, 1), (2, 0), (4, 2), (4, 7)]
    center = 250
    radius = 200  # 40% of image size
    angles = np.linspace(0, 2*np.pi, num_sides + 1)[:-1]
    
    vertices = []
    for angle in angles:
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        vertices.append((x, y))
    img_pil = Image.new('RGB', (500, 500), color='white')
    draw = ImageDraw.Draw(img_pil)
    
    draw.polygon(vertices, outline='black', fill=None, width=line_width)
    for start, end in diagonals:
        draw.line([vertices[start], vertices[end]], fill='black', width=line_width)
    save_path = f"polygon.png"
    img_pil.save(save_path)
    return save_path

if __name__ == "__main__":
    main()
