# Compositional Math Problem Settings
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/allenai/omega-compositional)

This directory contains compositional mathematical problem settings that combine different mathematical domains to evaluate model performance on complex, multi-domain reasoning tasks.

## Overview

Each compositional setting combines two distinct mathematical domains from the training datasets to create evaluation scenarios that test the model's ability to apply knowledge across different mathematical areas simultaneously.

## Compositional Settings

### Setting 1: Algebra (Polynomial Roots) + Arithmetic (GCD)

**Training Datasets:**
- [Algebra - Polynomial Roots](https://huggingface.co/datasets/sunyiyou/math_algebra_polynomial_roots_7B_train)
- [Arithmetic - GCD](https://huggingface.co/datasets/sunyiyou/math_arithmetic_gcd_7B_train)

**Evaluation Dataset:**
- [Polynomial GCD Composition](https://huggingface.co/datasets/sunyiyou/math_comp_polynomial_gcd)

### Setting 2: Geometry (Rotation) + Combinatorics (Pattern Matching)

**Training Datasets:**
- [Geometry - Polygon Rotation](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_rotation_7B_train)
- [Combinatorics - Pattern Matching](https://huggingface.co/datasets/sunyiyou/math_combinatory_pattern_matching_7B_train)

**Evaluation Dataset:**
- [N-Gon Composition](https://huggingface.co/datasets/sunyiyou/math_comp_n_gon)

### Setting 3: Geometry (Circle) + Algebra (Function Intersection)

**Training Datasets:**
- [Geometry - Circle](https://huggingface.co/datasets/sunyiyou/math_geometry_circle_7B_train)
- [Algebra - Function Intersection](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_7B_train)

**Evaluation Dataset:**
- [Circles Algebra Composition](https://huggingface.co/datasets/sunyiyou/math_comp_circles_algebra)

### Setting 4: Combinatorics (Probability) + Algebra (Function Intersection)

**Training Datasets:**
- [Combinatorics - Probability (No Fixed Letter)](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_train)
- [Algebra - Function Intersection](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_train)

**Evaluation Dataset:**
- [Parametric Intersection Composition](https://huggingface.co/datasets/sunyiyou/math_comp_parametric_intersection)

### Setting 5: Arithmetic (Matrix Rank) + Combinatorics (Probability)

**Training Datasets:**
- [Arithmetic - Matrix Rank](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_rank_7B_train)
- [Combinatorics - Probability (No Fixed Letter)](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_7B_train)

**Evaluation Dataset:**
- [Matrix Rank Composition](https://huggingface.co/datasets/sunyiyou/math_comp_matrix_rank)

### Setting 6: Geometry (Polygon Color) + Combinatorics (Probability)

**Training Datasets:**
- [Geometry - Polygon Color](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_color_7B_train)
- [Combinatorics - Probability (No Fixed Letter)](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_7B_train)

**Evaluation Dataset:**
- [Vertex Color Composition](https://huggingface.co/datasets/sunyiyou/math_comp_vertex_color)

### Setting 7: Logic (Grid Chip) + Combinatorics (Probability)

**Training Datasets:**
- [Logic - Grid Chip](https://huggingface.co/datasets/sunyiyou/math_logic_puzzles_grid_chip_7B_train)
- [Combinatorics - Probability (No Fixed Letter)](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_7B_train)

**Evaluation Dataset:**
- [Grid Chips Composition](https://huggingface.co/datasets/sunyiyou/math_comp_grid_chips)


### More is comming!