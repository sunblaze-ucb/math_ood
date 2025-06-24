# Explorative Math Problem Settings
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/allenai/omega-explorative)

This directory contains explorative mathematical problem settings that assess whether a model can **faithfully extend** a single reasoning strategy beyond the range of complexities seen during training.

## Overview

Exploratory generalization assesses whether a model can faithfully extend a single reasoning strategy beyond the range of complexities seen during training. Concretely, the model is exposed to problems drawn from one template, all lying within a "low-complexity" regime, and is then evaluated on **harder** instances from the same family. This axis probes robustness: does the model generalize the same algorithm to higher complexity problems? or does it merely memorize solutions at a fixed difficulty level?

Each explorative setting consists of:
- **Training Dataset**: Low-complexity problems from a specific mathematical domain
- **Test In-Distribution**: Problems of similar complexity to training data
- **Test Out-of-Distribution**: Higher complexity problems from the same domain/template

## Explorative Settings by Mathematical Domain

### Algebra - Function Analysis

#### Function Area Calculation
**Training Dataset:**
- [Function Area - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_area_7B_train)

**Evaluation Datasets:**
- [Function Area - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_area_7B_test_in)
- [Function Area - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_area_7B_test_out)

#### Function Derivative Sign Analysis
**Training Dataset:**
- [Function Derivative Sign - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_derivative_sign_7B_train)

**Evaluation Datasets:**
- [Function Derivative Sign - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_derivative_sign_7B_test_in)
- [Function Derivative Sign - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_derivative_sign_7B_test_out)

#### Function Extrema Detection
**Training Dataset:**
- [Function Extrema - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_extrema_7B_train)

**Evaluation Datasets:**
- [Function Extrema - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_extrema_7B_test_in)
- [Function Extrema - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_extrema_7B_test_out)

#### Function Extrema Coordinates
**Training Dataset:**
- [Function Extrema Coordinates - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_extrema_coords_7B_train)

**Evaluation Datasets:**
- [Function Extrema Coordinates - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_extrema_coords_7B_test_in)
- [Function Extrema Coordinates - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_extrema_coords_7B_test_out)

#### Function Intersection Analysis
**Training Dataset:**
- [Function Intersection - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_7B_train)

**Evaluation Datasets:**
- [Function Intersection - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_7B_test_in)
- [Function Intersection - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_7B_test_out)

#### Function Intersection Coordinates
**Training Dataset:**
- [Function Intersection Coordinates - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_coords_7B_train)

**Evaluation Datasets:**
- [Function Intersection Coordinates - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_coords_7B_test_in)
- [Function Intersection Coordinates - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_intersection_coords_7B_test_out)

#### Function Zeros Analysis
**Training Dataset:**
- [Function Zeros - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_func_zeros_7B_train)

**Evaluation Datasets:**
- [Function Zeros - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_zeros_7B_test_in)
- [Function Zeros - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_func_zeros_7B_test_out)

### Algebra - Equation Solving

#### Linear Equation Solving
**Training Dataset:**
- [Linear Equation - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_linear_equation_7B_train)

**Evaluation Datasets:**
- [Linear Equation - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_linear_equation_7B_test_in)
- [Linear Equation - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_linear_equation_7B_test_out)

#### Polynomial Roots
**Training Dataset:**
- [Polynomial Roots - Training](https://huggingface.co/datasets/sunyiyou/math_algebra_polynomial_roots_7B_train)

**Evaluation Datasets:**
- [Polynomial Roots - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_polynomial_roots_7B_test_in)
- [Polynomial Roots - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_algebra_polynomial_roots_7B_test_out)

### Arithmetic - Number Theory and Operations

#### Greatest Common Divisor (GCD)
**Training Dataset:**
- [GCD - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_gcd_7B_train)

**Evaluation Datasets:**
- [GCD - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_gcd_7B_test_in)
- [GCD - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_gcd_7B_test_out)

#### Prime Factorization
**Training Dataset:**
- [List Prime Factors - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_list_prime_factors_7B_train)

**Evaluation Datasets:**
- [List Prime Factors - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_list_prime_factors_7B_test_in)
- [List Prime Factors - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_list_prime_factors_7B_test_out)

#### Mixed Arithmetic Operations
**Training Dataset:**
- [Mixed Arithmetic - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_mixed_7B_train)

**Evaluation Datasets:**
- [Mixed Arithmetic - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_mixed_7B_test_in)
- [Mixed Arithmetic - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_mixed_7B_test_out)

### Arithmetic - Matrix Operations

#### Matrix Determinant
**Training Dataset:**
- [Matrix Determinant - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_determinant_7B_train)

**Evaluation Datasets:**
- [Matrix Determinant - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_determinant_7B_test_in)
- [Matrix Determinant - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_determinant_7B_test_out)

#### Matrix Eigenvalues
**Training Dataset:**
- [Matrix Eigenvalues - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_eigenvalues_7B_train)

**Evaluation Datasets:**
- [Matrix Eigenvalues - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_eigenvalues_7B_test_in)
- [Matrix Eigenvalues - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_eigenvalues_7B_test_out)

#### Matrix Inverse
**Training Dataset:**
- [Matrix Inverse - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_inverse_7B_train)

**Evaluation Datasets:**
- [Matrix Inverse - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_inverse_7B_test_in)
- [Matrix Inverse - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_inverse_7B_test_out)

#### Matrix Multiplication
**Training Dataset:**
- [Matrix Multiplication - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_multiplication_7B_train)

**Evaluation Datasets:**
- [Matrix Multiplication - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_multiplication_7B_test_in)
- [Matrix Multiplication - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_multiplication_7B_test_out)

#### Matrix Power
**Training Dataset:**
- [Matrix Power - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_power_7B_train)

**Evaluation Datasets:**
- [Matrix Power - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_power_7B_test_in)
- [Matrix Power - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_power_7B_test_out)

#### Matrix Rank
**Training Dataset:**
- [Matrix Rank - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_rank_7B_train)

**Evaluation Datasets:**
- [Matrix Rank - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_rank_7B_test_in)
- [Matrix Rank - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_rank_7B_test_out)

#### Matrix SVD (Singular Value Decomposition)
**Training Dataset:**
- [Matrix SVD - Training](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_svd_7B_train)

**Evaluation Datasets:**
- [Matrix SVD - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_svd_7B_test_in)
- [Matrix SVD - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_arithmetic_matrix_svd_7B_test_out)

### Combinatorics - Probability and Pattern Analysis

#### Distribution Analysis
**Training Dataset:**
- [Distribution - Training](https://huggingface.co/datasets/sunyiyou/math_combinatory_distribution_7B_train)

**Evaluation Datasets:**
- [Distribution - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_distribution_7B_test_in)
- [Distribution - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_distribution_7B_test_out)

#### Pattern Matching
**Training Dataset:**
- [Pattern Matching - Training](https://huggingface.co/datasets/sunyiyou/math_combinatory_pattern_matching_7B_train)

**Evaluation Datasets:**
- [Pattern Matching - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_pattern_matching_7B_test_in)
- [Pattern Matching - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_pattern_matching_7B_test_out)

#### Probability - At Least N Specific Fixed Points
**Training Dataset:**
- [Probability At Least N Specific Fixed - Training](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_at-least-n-specific-fixed_7B_train)

**Evaluation Datasets:**
- [Probability At Least N Specific Fixed - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_at-least-n-specific-fixed_7B_test_in)
- [Probability At Least N Specific Fixed - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_at-least-n-specific-fixed_7B_test_out)

#### Probability - Exactly N Specific Fixed Points
**Training Dataset:**
- [Probability Exactly N Specific Fixed - Training](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_exactly-n-specific-fixed_7B_train)

**Evaluation Datasets:**
- [Probability Exactly N Specific Fixed - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_exactly-n-specific-fixed_7B_test_in)
- [Probability Exactly N Specific Fixed - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_exactly-n-specific-fixed_7B_test_out)

#### Probability - No Fixed Points
**Training Dataset:**
- [Probability No Fixed Points - Training](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-fixed-points_7B_train)

**Evaluation Datasets:**
- [Probability No Fixed Points - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-fixed-points_7B_test_in)
- [Probability No Fixed Points - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-fixed-points_7B_test_out)

#### Probability - No Specific Letter Fixed
**Training Dataset:**
- [Probability No Specific Letter Fixed - Training](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_7B_train)

**Evaluation Datasets:**
- [Probability No Specific Letter Fixed - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_7B_test_in)
- [Probability No Specific Letter Fixed - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_combinatory_probability_no-specific-letter-fixed_7B_test_out)

### Geometry - Spatial and Shape Analysis

#### Basic Geometry
**Training Dataset:**
- [Basic Geometry - Training](https://huggingface.co/datasets/sunyiyou/math_geometry_basic_7B_train)

**Evaluation Datasets:**
- [Basic Geometry - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_basic_7B_test_in)
- [Basic Geometry - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_basic_7B_test_out)

#### Circle Geometry
**Training Dataset:**
- [Circle Geometry - Training](https://huggingface.co/datasets/sunyiyou/math_geometry_circle_7B_train)

**Evaluation Datasets:**
- [Circle Geometry - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_circle_7B_test_in)
- [Circle Geometry - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_circle_7B_test_out)

#### Perpendicular Intersection
**Training Dataset:**
- [Perpendicular Intersection - Training](https://huggingface.co/datasets/sunyiyou/math_geometry_perpendicular_intersection_7B_train)

**Evaluation Datasets:**
- [Perpendicular Intersection - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_perpendicular_intersection_7B_test_in)
- [Perpendicular Intersection - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_perpendicular_intersection_7B_test_out)

#### Polygon Analysis
**Training Dataset:**
- [Polygon Analysis - Training](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_7B_train)

**Evaluation Datasets:**
- [Polygon Analysis - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_7B_test_in)
- [Polygon Analysis - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_7B_test_out)

#### Polygon Rotation
**Training Dataset:**
- [Polygon Rotation - Training](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_rotation_7B_train)

**Evaluation Datasets:**
- [Polygon Rotation - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_rotation_7B_test_in)
- [Polygon Rotation - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_polygon_rotation_7B_test_out)

#### Triangle Geometry
**Training Dataset:**
- [Triangle Geometry - Training](https://huggingface.co/datasets/sunyiyou/math_geometry_triangle_7B_train)

**Evaluation Datasets:**
- [Triangle Geometry - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_triangle_7B_test_in)
- [Triangle Geometry - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_geometry_triangle_7B_test_out)

### Logic - Grid-Based and Constraint Reasoning

#### Gridworld Blocked
**Training Dataset:**
- [Gridworld Blocked - Training](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_blocked_7B_train)

**Evaluation Datasets:**
- [Gridworld Blocked - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_blocked_7B_test_in)
- [Gridworld Blocked - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_blocked_7B_test_out)

#### Gridworld Knight Move
**Training Dataset:**
- [Gridworld Knight Move - Training](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_knight_move_7B_train)

**Evaluation Datasets:**
- [Gridworld Knight Move - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_knight_move_7B_test_in)
- [Gridworld Knight Move - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_knight_move_7B_test_out)

#### Gridworld Rook Move
**Training Dataset:**
- [Gridworld Rook Move - Training](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_rookmove_7B_train)

**Evaluation Datasets:**
- [Gridworld Rook Move - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_rookmove_7B_test_in)
- [Gridworld Rook Move - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_gridworld_rookmove_7B_test_out)

#### Zebra Logic
**Training Dataset:**
- [Zebra Logic - Training](https://huggingface.co/datasets/sunyiyou/math_logic_zebralogic_7B_train)

**Evaluation Datasets:**
- [Zebra Logic - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_zebralogic_7B_test_in)
- [Zebra Logic - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_logic_zebralogic_7B_test_out)

### Number Theory - Advanced Mathematical Structures

#### Prime Mod
**Training Dataset:**
- [Prime Mod - Training](https://huggingface.co/datasets/sunyiyou/math_numbertheory_lte_qr_7B_train)

**Evaluation Datasets:**
- [Prime Mod - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_numbertheory_lte_qr_7B_test_in)
- [Prime Mod - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_numbertheory_lte_qr_7B_test_out)

#### Triple_Count

**Training Dataset:**
- [Triple Count - Training](https://huggingface.co/datasets/sunyiyou/math_numbertheory_ordered_lte_7B_train)

**Evaluation Datasets:**
- [Triple Count - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_numbertheory_ordered_lte_7B_test_in)
- [Triple Count - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_numbertheory_ordered_lte_7B_test_out)

#### Digital Sum
**Training Dataset:**
- [Digital Sum - Training](https://huggingface.co/datasets/sunyiyou/math_numbertheory_qr_sum_7B_train)

**Evaluation Datasets:**
- [Digital Sum - Test In-Distribution](https://huggingface.co/datasets/sunyiyou/math_numbertheory_qr_sum_7B_test_in)
- [Digital Sum - Test Out-of-Distribution](https://huggingface.co/datasets/sunyiyou/math_numbertheory_qr_sum_7B_test_out)