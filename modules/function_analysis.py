"""
Function analysis module containing various 2D function problem generators.
This module generates problems related to function intersections, extrema,
integration, differentiation, and other analytical properties.

The module supports generating problems with different difficulty levels (1-5),
where higher levels use more complex functions and problem types:
    - Level 1: Simple linear functions
    - Level 2: Quadratic and absolute value functions
    - Level 3: Trigonometric functions and simple composites
    - Level 4: Exponential, logarithmic functions and moderate composites
    - Level 5: Rational functions and complex composites

Each function generator returns a tuple of (function, expression, difficulty_info),
where difficulty_info is a dictionary containing metadata about the function's complexity.
"""

import numpy as np
import random
import argparse
import math
from collections import namedtuple

from pycparser.c_ast import While
from scipy import optimize, integrate
import sympy as sp
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Simple Problem container with question, answer, plot data, and difficulty information
Problem = namedtuple('Problem', ('question', 'answer', 'plot_data', 'difficulty'))

# --- Function Generation Utilities ---

def generate_linear_function(min_coef=-5, max_coef=5):
    """Generate a linear function f(x) = ax + b with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:  # Avoid horizontal lines
        a = random.choice([-1, 1])
    b = random.randint(min_coef, max_coef)
    
    def f(x):
        return a * x + b
    
    expr = f"{a}x"
    if b > 0:
        expr += f" + {b}"
    elif b < 0:
        expr += f" - {abs(b)}"
    
    difficulty_info = {
        "type": "linear",
        "difficulty": 1
    }
    
    return f, expr, difficulty_info

def generate_quadratic_function(min_coef=-5, max_coef=5):
    """Generate a quadratic function f(x) = ax^2 + bx + c with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:  # Avoid linear functions
        a = random.choice([-1, 1])
    b = random.randint(min_coef, max_coef)
    c = random.randint(min_coef, max_coef)
    
    def f(x):
        return a * x**2 + b * x + c
    
    expr = f"{a}x^2"
    if b > 0:
        expr += f" + {b}x"
    elif b < 0:
        expr += f" - {abs(b)}x"
    
    if c > 0:
        expr += f" + {c}"
    elif c < 0:
        expr += f" - {abs(c)}"
    
    difficulty_info = {
        "type": "quadratic",
        "difficulty": 2
    }
    
    return f, expr, difficulty_info

def generate_sin_function(min_coef=-3, max_coef=3, min_freq=1, max_freq=3):
    """Generate a sine function f(x) = a * sin(b*x + c) + d with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(min_freq, max_freq)
    c = random.randint(-3, 3)
    d = random.randint(min_coef, max_coef)
    
    def f(x):
        return a * np.sin(b * np.pi * x + c) + d
    
    expr = f"{a}sin({b}πx"
    if c > 0:
        expr += f" + {c}"
    elif c < 0:
        expr += f" - {abs(c)}"
    expr += ")"
    
    if d > 0:
        expr += f" + {d}"
    elif d < 0:
        expr += f" - {abs(d)}"
    
    difficulty_info = {
        "type": "sin",
        "difficulty": 4,
        "frequency": b
    }
    
    return f, expr, difficulty_info

def generate_cos_function(min_coef=-3, max_coef=3, min_freq=1, max_freq=3):
    """Generate a cosine function f(x) = a * cos(b*x + c) + d with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(min_freq, max_freq)
    c = random.randint(-3, 3)
    d = random.randint(min_coef, max_coef)
    
    def f(x):
        return a * np.cos(b * np.pi * x + c) + d
    
    expr = f"{a}cos({b}πx"
    if c > 0:
        expr += f" + {c}"
    elif c < 0:
        expr += f" - {abs(c)}"
    expr += ")"
    
    if d > 0:
        expr += f" + {d}"
    elif d < 0:
        expr += f" - {abs(d)}"
    
    difficulty_info = {
        "type": "cos",
        "difficulty": 4,
        "frequency": b
    }
    
    return f, expr, difficulty_info

def generate_abs_function(min_coef=-3, max_coef=3):
    """Generate an absolute value function f(x) = a * |x - b| + c with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(-3, 3)
    c = random.randint(min_coef, max_coef)
    
    def f(x):
        return a * abs(x - b) + c
    
    expr = f"{a}|x"
    if b > 0:
        expr += f" - {b}"
    elif b < 0:
        expr += f" + {abs(b)}"
    expr += "|"
    
    if c > 0:
        expr += f" + {c}"
    elif c < 0:
        expr += f" - {abs(c)}"
    
    difficulty_info = {
        "type": "abs",
        "difficulty": 3
    }
    
    return f, expr, difficulty_info

def generate_exp_function(min_coef=-3, max_coef=3):
    """Generate an exponential function f(x) = a * e^(b*x) + c with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(-2, 2)
    if b == 0:
        b = random.choice([-1, 1])
    c = random.randint(min_coef, max_coef)
    
    def f(x):
        # Add safety check to prevent overflow in exponential function
        safe_exp = np.exp(np.clip(b * x, -100, 100))  # Clip to safe range
        return a * safe_exp + c
    
    expr = f"{a}e^({b}x)"
    if c > 0:
        expr += f" + {c}"
    elif c < 0:
        expr += f" - {abs(c)}"
    
    difficulty_info = {
        "type": "exp",
        "difficulty": 3
    }
    
    return f, expr, difficulty_info

def generate_log_function(min_coef=-3, max_coef=3):
    """Generate a logarithmic function f(x) = a * ln(bx + c) + d with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(1, 3)  # Positive to ensure domain validity
    c = random.randint(1, 5)  # Ensure bx + c > 0 for x ≥ 0
    d = random.randint(min_coef, max_coef)
    
    def f(x):
        # Handle domain issues with a small epsilon to prevent errors
        arg = b * x + c
        # Ensure positive argument for logarithm
        if np.isscalar(arg):
            arg = max(arg, 1e-10)  # Ensure arg is positive
        else:
            arg = np.maximum(arg, 1e-10)  # Use np.maximum for arrays
        return a * np.log(arg) + d
    
    expr = f"{a}ln({b}x"
    if c > 0:
        expr += f" + {c}"
    elif c < 0:
        expr += f" - {abs(c)}"
    expr += ")"
    
    if d > 0:
        expr += f" + {d}"
    elif d < 0:
        expr += f" - {abs(d)}"
    
    difficulty_info = {
        "type": "log",
        "difficulty": 3
    }
    
    return f, expr, difficulty_info

def generate_rational_function(min_coef=-3, max_coef=3):
    """Generate a rational function f(x) = (ax + b)/(cx + d) with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(min_coef, max_coef)
    c = random.randint(min_coef, max_coef)
    if c == 0:
        c = random.choice([-2, -1, 1, 2])
    d = random.randint(min_coef, max_coef)
    if d == -c:  # Avoid division by zero at x = 1
        d += 1
    
    def f(x):
        # Handle denominator to prevent division by zero
        numerator = a * x + b
        denominator = c * x + d
        
        # Add epsilon only where necessary
        if np.isscalar(denominator):
            # For scalar inputs
            if abs(denominator) < 1e-10:
                denominator = 1e-10 if denominator >= 0 else -1e-10
        else:
            # For array inputs
            small_denom = np.abs(denominator) < 1e-10
            if np.any(small_denom):
                # Replace only small values with epsilon
                sign = np.sign(denominator)
                sign[sign == 0] = 1  # Handle exact zeros
                denominator = np.where(small_denom, 1e-10 * sign, denominator)
        
        return numerator / denominator
    
    # Create expression string
    num = f"{a}x"
    if b > 0:
        num += f" + {b}"
    elif b < 0:
        num += f" - {abs(b)}"
    
    den = f"{c}x"
    if d > 0:
        den += f" + {d}"
    elif d < 0:
        den += f" - {abs(d)}"
    
    expr = f"({num})/({den})"
    
    difficulty_info = {
        "type": "rational",
        "difficulty": 4
    }
    
    return f, expr, difficulty_info

def generate_composite_function(num_components=None):
    """
    Generate a composite function by combining simpler functions
    
    Args:
        num_components: Number of functions to compose (None for random 2-3)
    
    Returns:
        tuple: (function, expression, difficulty_info)
        where difficulty_info is a dict containing metadata about the function
    """
    # Choose the number of components (2 or 3 if not specified)
    if num_components is None:
        num_components = random.randint(2, 3)
    
    # Available function generators
    elementary_generators = [
        generate_linear_function,
        generate_quadratic_function,
        generate_abs_function,
        generate_rational_function
    ]
    
    advanced_generators = [
        generate_sin_function,
        generate_cos_function,
        generate_exp_function,
        # generate_log_function,  # Can cause domain issues in composition
    ]
    
    # Include at most one advanced generator (sin/cos/exp)
    chosen_generators = []
    
    # Decide whether to include an advanced generator
    include_advanced = random.choice([True, False])
    
    if include_advanced and num_components > 1:
        # Choose one advanced generator
        advanced_gen = random.choice(advanced_generators)
        chosen_generators.append(advanced_gen)
        
        # Fill the rest with elementary generators
        remaining_needed = num_components - 1
        chosen_generators.extend(np.random.choice(elementary_generators, remaining_needed))
    else:
        # All elementary generators
        # chosen_generators = random.sample(elementary_generators, num_components)
        chosen_generators = np.random.choice(elementary_generators, num_components)
    
    # Shuffle to randomize position of the advanced function in the composition
    random.shuffle(chosen_generators)
    
    # Generate the component functions
    components = [gen() for gen in chosen_generators]
    
    # Unpack the components - each component is now (func, expr, info)
    funcs = []
    exprs = []
    infos = []
    for f, expr, info in components:
        funcs.append(f)
        exprs.append(expr)
        infos.append(info)
    
    # Create the composite function
    def composite_func(x):
        result = funcs[0](x)
        for func in funcs[1:]:
            result = func(result)
        return result
    
    # Create the expression string
    composite_expr = exprs[-1]
    for expr in reversed(exprs[:-1]):
        composite_expr = composite_expr.replace("x", f"({expr})")
    
    # Create difficulty info dictionary
    difficulty_info = {
        "type": "composite",
        "num_components": num_components,
        "component_types": [info.get('type', 'unknown') for info in infos],
        "component_difficulties": [info.get('difficulty', -1) for info in infos]
    }
    
    # Calculate overall difficulty based on component difficulties
    if infos:
        difficulty_info["difficulty"] = sum(info.get('difficulty', -1) for info in infos)
    
    return composite_func, composite_expr, difficulty_info

def generate_random_function(difficulty_level=None):
    """
    Generate a random function of one of the available types based on difficulty level
    without using composites.
    
    Args:
        difficulty_level: Optional integer (1-5) to control difficulty
                        1: Linear
                        2: Quadratic, Absolute value
                        3: Exponential, Logarithmic
                        4: Trigonometric
                        5: Rational
    
    Returns:
        tuple: (function, expression, difficulty_info)
    """
    # Map difficulty levels to function generators
    difficulty_map = {
        1: [generate_linear_function],
        2: [generate_quadratic_function, generate_abs_function],
        3: [generate_exp_function, generate_log_function],
        4: [generate_sin_function, generate_cos_function],
        5: [generate_rational_function]
    }
    
    if difficulty_level is not None:
        # Validate difficulty level
        if difficulty_level < 1:
            difficulty_level = 1
        elif difficulty_level > 5:
            difficulty_level = 5
        
        # Choose from appropriate generators for this difficulty level
        generator = random.choice(difficulty_map[difficulty_level])
        return generator()
    else:
        # No difficulty specified, select any non-composite function type
        function_generators = [
            generate_linear_function,
            generate_quadratic_function,
            generate_sin_function,
            generate_cos_function,
            generate_abs_function,
            generate_exp_function,
            generate_log_function,
            generate_rational_function
        ]
        
        generator = random.choice(function_generators)
        return generator()

def find_intersections(f1, f2, x_min=-10, x_max=10, num_points=5000):
    """
    Find the intersection points of two functions f1 and f2, including tangent points
    
    Args:
        f1: First function
        f2: Second function
        x_min: Minimum x value to search
        x_max: Maximum x value to search
        num_points: Number of points to sample for initial intersection detection
        
    Returns:
        List of x-values where the functions intersect or are tangent
    """
    # Create a function representing the difference
    def diff(x):
        return f1(x) - f2(x)
    
    # Sample points to look for sign changes
    x_samples = np.linspace(x_min, x_max, num_points)
    y_diff = diff(x_samples)
    
    # Find sign changes (crossing intersections)
    sign_changes = np.where(np.diff(np.signbit(y_diff)))[0]
    
    # For each sign change, find the precise root
    intersections = []
    for i in sign_changes:
        x_left, x_right = x_samples[i], x_samples[i+1]
        try:
            root, = optimize.fsolve(diff, (x_left + x_right) / 2)
            # Check if the root is valid and within bounds
            if x_min <= root <= x_max and abs(diff(root)) < 1e-5:
                # Round to 4 decimal places
                intersections.append(round(root, 2))
        except:
            continue
    
    # Find tangent points (where functions touch but don't cross)
    # First, create a function to minimize (absolute difference between functions)
    def abs_diff(x):
        return abs(diff(x))

    # Find potential tangent points by looking for places where diff(x) is close to zero
    potential_tangent_points = []
    threshold = 1e-3  # Threshold for "close to zero"

    # Find all points where |diff(x)| is below threshold
    near_zero_indices = np.where(np.abs(y_diff) < threshold)[0]

    # Group adjacent indices to avoid redundant searches
    if len(near_zero_indices) > 0:
        groups = []
        current_group = [near_zero_indices[0]]

        for i in range(1, len(near_zero_indices)):
            if near_zero_indices[i] - near_zero_indices[i-1] <= 2:  # Consider adjacent if within 2 indices
                current_group.append(near_zero_indices[i])
            else:
                groups.append(current_group)
                current_group = [near_zero_indices[i]]

        if current_group:
            groups.append(current_group)

        # Take the middle index from each group as a potential tangent point
        for group in groups:
            mid_idx = group[len(group) // 2]
            potential_tangent_points.append(x_samples[mid_idx])

    # Add extrema of the diff function as potential tangent points
    # Sample the derivative of diff to find where it changes sign
    try:
        def diff_prime(x):
            h = 1e-5
            try:
                # Calculate diff at neighborhood points
                diff_plus = diff(x + h)
                diff_minus = diff(x - h)
                
                # Check for invalid values
                if not np.isfinite(diff_plus) or not np.isfinite(diff_minus):
                    return 0  # Return 0 for invalid values
                    
                return (diff_plus - diff_minus) / (2 * h)
            except:
                return 0  # Return 0 on error

        x_deriv_samples = np.linspace(x_min, x_max, num_points)
        y_deriv = np.array([diff_prime(x) for x in x_deriv_samples])
        deriv_sign_changes = np.where(np.diff(np.signbit(y_deriv)))[0]

        for i in deriv_sign_changes:
            x_point = (x_deriv_samples[i] + x_deriv_samples[i+1]) / 2
            potential_tangent_points.append(x_point)

    except:
        pass  # If derivative calculation fails, continue with other methods

    # Search for tangent points around each potential point
    for x_start in potential_tangent_points:
        try:
            result = optimize.minimize(abs_diff, x_start,
                                      bounds=[(max(x_min, x_start - 0.5), min(x_max, x_start + 0.5))])

            # If the minimum is close to zero and not already found
            if result.success and abs(diff(result.x[0])) < 1e-5:
                tangent_point = round(result.x[0], 2)

                # Check if this is a true tangent point (not a crossing)
                # by checking nearby points on both sides
                epsilon = 1e-6
                left_sign = np.sign(diff(result.x[0] - epsilon))
                right_sign = np.sign(diff(result.x[0] + epsilon))

                # For tangent points, the sign doesn't change
                if left_sign == right_sign and tangent_point not in intersections:
                    intersections.append(tangent_point)
        except:
            continue
    
    # Remove duplicates and sort
    intersections = sorted(list(set(intersections)))
    return intersections

def find_extrema(f, x_min=-10, x_max=10, num_points=1000):
    """
    Find the local extrema (minima and maxima) of a function
    
    Args:
        f: The function to analyze
        x_min: Minimum x value to search
        x_max: Maximum x value to search
        num_points: Number of points to sample for initial extrema detection
        
    Returns:
        List of tuples (x, y, type) where type is 'min' or 'max'
    """
    # Define the derivative (numerically)
    def f_prime(x):
        h = 1e-5
        try:
            # Calculate function values at neighborhood points
            f_plus = f(x + h)
            f_minus = f(x - h)
            
            # Check for invalid values
            if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                return 0  # Return 0 for invalid values
            
            return (f_plus - f_minus) / (2 * h)
        except:
            return 0  # Return 0 on error
    
    # Sample points to look for sign changes in the derivative
    x_samples = np.linspace(x_min, x_max, num_points)
    y_prime = np.array([f_prime(x) for x in x_samples])
    
    # Find sign changes in the derivative
    sign_changes = np.where(np.diff(np.signbit(y_prime)))[0]
    
    extrema = []
    for i in sign_changes:
        x_left, x_right = x_samples[i], x_samples[i+1]
        try:
            # Find where the derivative is zero
            root, = optimize.fsolve(f_prime, (x_left + x_right) / 2)
            
            # Check if the root is valid and within bounds
            if x_min <= root <= x_max and abs(f_prime(root)) < 1e-5:
                # Determine if it's a minimum or maximum
                # Second derivative test
                h = 1e-5
                second_deriv = (f_prime(root + h) - f_prime(root - h)) / (2 * h)
                
                extremum_type = 'min' if second_deriv > 0 else 'max'
                # Round x to 2 decimal places and y to 2 decimal places
                extrema.append((round(root, 2), round(f(root), 2), extremum_type))
        except:
            continue
    
    return extrema

def create_function_plot(functions, labels, x_min=-10, x_max=10, points=None):
    """
    Create a plot of the functions
    
    Args:
        functions: List of function objects
        labels: List of function labels
        x_min: Minimum x value
        x_max: Maximum x value
        points: Optional list of special points to mark (x, y, label)
        
    Returns:
        Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 6))
    
    x = np.linspace(x_min, x_max, 1000)
    
    # Track all valid y values to determine appropriate plot limits
    all_y_values = []
    
    for func, label in zip(functions, labels):
        try:
            y = np.array([func(xi) for xi in x])
            # Filter out infinities and NaNs
            mask = np.isfinite(y)
            plt.plot(x[mask], y[mask], label=f"f(x) = {label}")
            all_y_values.extend(y[mask].tolist())
        except:
            # If there's an error, try to plot point by point
            valid_x = []
            valid_y = []
            for xi in x:
                try:
                    yi = func(xi)
                    if np.isfinite(yi) and -1000 < yi < 1000:  # Filter extreme values
                        valid_x.append(xi)
                        valid_y.append(yi)
                except:
                    continue
            if valid_x:
                plt.plot(valid_x, valid_y, label=f"f(x) = {label}")
                all_y_values.extend(valid_y)
    
    # Plot special points and add them to y-values for limit calculation
    if points:
        point_y_values = []
        for x, y, point_label in points:
            plt.plot(x, y, 'ro', markersize=6)
            plt.annotate(point_label, (x, y), xytext=(5, 5), textcoords='offset points')
            point_y_values.append(y)
        all_y_values.extend(point_y_values)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Set dynamically adjusted y-limits based on the function values and intersection points
    if all_y_values:
        filtered_y = [y for y in all_y_values if -100 < y < 100]  # Filter out extreme values
        if filtered_y:
            # Calculate appropriate y limits
            y_min, y_max = min(filtered_y), max(filtered_y)
            # Add padding (15% of the range on each side)
            y_range = y_max - y_min
            padding = max(y_range * 0.15, 1)  # At least 1 unit of padding
            plt.ylim(y_min - padding, y_max + padding)
        else:
            # Fallback to default limits if no valid values after filtering
            plt.ylim(-100, 100)
    else:
        # Fallback to default limits if no y values were collected
        plt.ylim(-100, 100)
    
    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 string
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str

# --- Problem Generator Functions ---

def generate_intersection_problem(composition_nums=None):
    """
    Generate a problem about finding the number of intersection points between two functions
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate two functions, potentially composite based on composition_nums
    if composition_nums is not None:
        f1, expr1, f1_info = generate_composite_function(composition_nums)
        f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))  # Slightly less complex second function
    else:
        # Generate random non-composite functions if composition_nums is None
        f1, expr1, f1_info = generate_random_function()
        f2, expr2, f2_info = generate_random_function()
    
    # Ensure the functions are different
    while expr1 == expr2:
        if composition_nums is not None:
            f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))
        else:
            f2, expr2, f2_info = generate_random_function()
    
    # Find intersections in the range [-10, 10]
    intersections = find_intersections(f1, f2)
    
    # Try multiple times to find functions with at least one intersection
    max_attempts = 5
    attempts = 0
    
    while len(intersections) == 0 and attempts < max_attempts:
        attempts += 1
        
        if composition_nums is not None:
            f1, expr1, f1_info = generate_composite_function(composition_nums)
            f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))
        else:
            f1, expr1, f1_info = generate_random_function()
            f2, expr2, f2_info = generate_random_function()
            
        # Ensure functions are different
        while expr1 == expr2:
            if composition_nums is not None:
                f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))
            else:
                f2, expr2, f2_info = generate_random_function()
                
        intersections = find_intersections(f1, f2)

    # Create plot data
    plot_data = create_function_plot(
        [f1, f2], 
        [expr1, expr2],
        points=[(x, f1(x), f"({x}, {round(f1(x), 2)})") for x in intersections]
    )
    
    # Create problem text
    templates = [
        f"How many points of intersection exist between the functions f(x) = {expr1} and g(x) = {expr2} in the range -10 ≤ x ≤ 10?",
        f"Find the number of times the graphs of f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10].",
        f"How many solutions does the equation {expr1} = {expr2} have for x in the interval [-10, 10]?",
        f"Determine the number of intersection points between the functions f(x) = {expr1} and g(x) = {expr2} where -10 ≤ x ≤ 10."
    ]
    
    question = random.choice(templates)
    answer = len(intersections)
    
    # Calculate overall difficulty based on function types and number of intersections
    # Use the higher of the two function difficulties as base
    base_difficulty = f1_info.get('difficulty', 3) + f2_info.get('difficulty', 3)
    
    # Adjust for number of intersections (more intersections = higher difficulty)
    intersection_factor = int(len(intersections) ** 0.75)
    
    # Calculate final difficulty (scale 1-5)
    final_difficulty = base_difficulty + intersection_factor
    
    # Create metadata about the problem
    difficulty_info = {
        "level": final_difficulty,
        "functions": [f1_info, f2_info],
        "num_intersections": len(intersections)
    }
    
    return Problem(question=question, answer=answer, plot_data=plot_data, difficulty=difficulty_info)

def generate_intersection_coordinates_problem(composition_nums=None):
    """
    Generate a problem about finding the coordinates of intersection points between two functions
    and returning a single integer value
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate two functions, potentially composite based on composition_nums
    if composition_nums is not None:
        f1, expr1, f1_info = generate_composite_function(composition_nums)
        f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))  # Slightly less complex second function
    else:
        # Generate random non-composite functions if composition_nums is None
        f1, expr1, f1_info = generate_random_function()
        f2, expr2, f2_info = generate_random_function()
    
    # Find intersections
    intersections = find_intersections(f1, f2)
    
    # Generate functions that have 1-3 intersections for simplicity
    attempts = 0
    max_attempts = 10
    
    while (len(intersections) < 1 or len(intersections) > 3) and attempts < max_attempts:
        if composition_nums is not None:
            f1, expr1, f1_info = generate_composite_function(composition_nums)
            f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))
        else:
            f1, expr1, f1_info = generate_random_function()
            f2, expr2, f2_info = generate_random_function()
        
        # Find intersections
        intersections = find_intersections(f1, f2)
        attempts += 1
    
    # If we couldn't find suitable functions after max attempts, 
    # use simpler functions to ensure we have intersections
    if attempts >= max_attempts:
        f1, expr1, f1_info = generate_linear_function()
        f2, expr2, f2_info = generate_quadratic_function()
        intersections = find_intersections(f1, f2)
    
    # Create plot data
    plot_data = create_function_plot(
        [f1, f2], 
        [expr1, expr2],
        points=[(x, f1(x), f"({x}, {round(f1(x), 2)})") for x in intersections]
    )
    
    # Determine what integer to ask for based on the number of intersections
    if len(intersections) == 1:
        # Ask for the rounded integer value of the only intersection point
        integer_answer = round(intersections[0])
        question_type = "single"
    else:
        # Choose a question type for multiple intersections
        question_types = ["leftmost", "rightmost", "sum", "average", "index"]
        question_type = random.choice(question_types)
        
        if question_type == "leftmost":
            integer_answer = round(min(intersections))
        elif question_type == "rightmost":
            integer_answer = round(max(intersections))
        elif question_type == "sum":
            integer_answer = round(sum(intersections))
        elif question_type == "average":
            integer_answer = round(sum(intersections) / len(intersections))
        elif question_type == "index":
            # Ask for a specific intersection by index
            idx = random.randint(0, len(intersections) - 1)
            sorted_intersections = sorted(intersections)
            integer_answer = round(sorted_intersections[idx])
            question_type = f"index_{idx}"
    
    # Create problem text based on question type
    if question_type == "single":
        templates = [
            f"Find the integer value closest to the x-coordinate where f(x) = {expr1} and g(x) = {expr2} intersect in the range -10 ≤ x ≤ 10.",
            f"At what integer value (rounded) do the functions f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10]?",
            f"Round to the nearest integer: what is the x-coordinate where the graphs of f(x) = {expr1} and g(x) = {expr2} meet in the interval [-10, 10]?"
        ]
    elif question_type == "leftmost":
        templates = [
            f"Round to the nearest integer: what is the leftmost x-coordinate where f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10]?",
            f"Find the integer closest to the smallest x-value where the functions f(x) = {expr1} and g(x) = {expr2} meet in the interval [-10, 10]."
        ]
    elif question_type == "rightmost":
        templates = [
            f"Round to the nearest integer: what is the rightmost x-coordinate where f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10]?",
            f"Find the integer closest to the largest x-value where the functions f(x) = {expr1} and g(x) = {expr2} meet in the interval [-10, 10]."
        ]
    elif question_type == "sum":
        templates = [
            f"Find the sum of all x-coordinates (rounded to the nearest integer) where f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10].",
            f"If you add up all the x-coordinates where the functions f(x) = {expr1} and g(x) = {expr2} meet in the interval [-10, 10] and round to the nearest integer, what do you get?"
        ]
    elif question_type == "average":
        templates = [
            f"Find the average of all x-coordinates (rounded to the nearest integer) where f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10].",
            f"What is the mean value of the x-coordinates (rounded to the nearest integer) where the functions f(x) = {expr1} and g(x) = {expr2} meet in the interval [-10, 10]?"
        ]
    elif question_type.startswith("index_"):
        idx = int(question_type.split("_")[1])
        ordinals = ["first", "second", "third"]
        if idx < len(ordinals):
            idx_desc = ordinals[idx]
        else:
            idx_desc = f"{idx+1}th"
            
        templates = [
            f"When ordered from left to right, what is the {idx_desc} x-coordinate (rounded to the nearest integer) where f(x) = {expr1} and g(x) = {expr2} intersect in the interval [-10, 10]?",
            f"Find the {idx_desc} intersection point (counting from left to right) between the functions f(x) = {expr1} and g(x) = {expr2} in the interval [-10, 10], rounded to the nearest integer."
        ]
    
    question = random.choice(templates)
    
    # Format the answer as an integer
    answer_str = str(integer_answer)
    
    # Calculate difficulty based on function types and precise coordinate finding
    # Start with the highest function difficulty
    base_difficulty = f1_info.get('difficulty', 3) + f2_info.get('difficulty', 3)
    
    # Finding coordinates is harder than counting
    coordinate_factor = 1
    
    # More extrema points increase difficulty
    extrema_factor = min(1, len(intersections) - 1) if len(intersections) > 1 else 0
    
    # Final difficulty
    final_difficulty = base_difficulty + coordinate_factor + extrema_factor
    
    difficulty_info = {
        "level": final_difficulty,
        "functions": [f1_info, f2_info],
        "num_intersections": len(intersections),
        "question_type": question_type
    }
    
    return Problem(question=question, answer=answer_str, plot_data=plot_data, difficulty=difficulty_info)

def generate_extrema_problem(composition_nums=None):
    """
    Generate a problem about finding extrema (local maxima and minima) of a function
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate a function with the specified number of compositions
    if composition_nums is not None:
        f, expr, f_info = generate_composite_function(composition_nums)
    else:
        # Generate a random function if composition_nums is None
        f, expr, f_info = generate_random_function()
    
    # Find extrema
    extrema = find_extrema(f)
    
    # Count the number of minima and maxima
    num_minima = sum(1 for _, _, type_ in extrema if type_ == 'min')
    num_maxima = sum(1 for _, _, type_ in extrema if type_ == 'max')
    
    # If we don't have any extrema, try again with a quadratic function (which has one extremum)
    while len(extrema) == 0:
        if composition_nums is not None:
            f, expr, f_info = generate_composite_function(composition_nums)
        else:
            # Generate a random function if composition_nums is None
            f, expr, f_info = generate_random_function()
        extrema = find_extrema(f)
        num_minima = sum(1 for _, _, type_ in extrema if type_ == 'min')
        num_maxima = sum(1 for _, _, type_ in extrema if type_ == 'max')
    
    # Create plot data with extrema points marked
    plot_points = []
    for x, y, type_ in extrema:
        label = "min" if type_ == 'min' else "max"
        plot_points.append((x, y, f"{label} ({x}, {y})"))
    
    plot_data = create_function_plot([f], [expr], points=plot_points)
    
    # Problem type: count extrema
    problem_types = ['count_all', 'count_minima', 'count_maxima']
    problem_type = random.choice(problem_types)
    
    if problem_type == 'count_all':
        templates = [
            f"How many local extrema (minima and maxima combined) does the function f(x) = {expr} have in the interval [-10, 10]?",
            f"Find the total number of local minima and maxima for f(x) = {expr} where -10 ≤ x ≤ 10.",
            f"Determine the total count of local extrema points for the function f(x) = {expr} in the interval [-10, 10]."
        ]
        answer = len(extrema)
    
    elif problem_type == 'count_minima':
        templates = [
            f"How many local minima does the function f(x) = {expr} have in the interval [-10, 10]?",
            f"Find the number of local minimum points for f(x) = {expr} where -10 ≤ x ≤ 10.",
            f"Determine the count of local minima for the function f(x) = {expr} in the interval [-10, 10]."
        ]
        answer = num_minima
    
    else:  # count_maxima
        templates = [
            f"How many local maxima does the function f(x) = {expr} have in the interval [-10, 10]?",
            f"Find the number of local maximum points for f(x) = {expr} where -10 ≤ x ≤ 10.",
            f"Determine the count of local maxima for the function f(x) = {expr} in the interval [-10, 10]."
        ]
        answer = num_maxima
    
    question = random.choice(templates)
    
    # Calculate difficulty based on function complexity and number of extrema
    base_difficulty = f_info.get('difficulty', 3)
    
    # More extrema points increase difficulty
    extrema_factor = min(2, len(extrema) - 1) if len(extrema) > 1 else 0
    
    # Final difficulty
    final_difficulty = base_difficulty + extrema_factor
    
    difficulty_info = {
        "level": final_difficulty,
        "functions": [f_info],
        "num_extrema": len(extrema),
        "num_minima": num_minima,
        "num_maxima": num_maxima,
        "problem_type": problem_type
    }
    
    return Problem(question=question, answer=answer, plot_data=plot_data, difficulty=difficulty_info)

def generate_extrema_coordinates_problem(composition_nums=None):
    """
    Generate a problem about finding coordinates of extrema of a function
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate a function with the specified number of compositions
    if composition_nums is not None:
        f, expr, f_info = generate_composite_function(composition_nums)
    else:
        # Generate a random function if composition_nums is None
        f, expr, f_info = generate_random_function()
    
    # Find extrema
    extrema = find_extrema(f)
    
    # Ensure we have at least one extremum
    while not extrema:
        if composition_nums is not None:
            f, expr, f_info = generate_composite_function(composition_nums)
        else:
            # Generate a random function if composition_nums is None
            f, expr, f_info = generate_random_function()
        extrema = find_extrema(f)
    
    # Create plot data with extrema points marked
    plot_points = []
    for x, y, type_ in extrema:
        label = "min" if type_ == 'min' else "max"
        plot_points.append((x, y, f"{label} ({x}, {y})"))
    
    plot_data = create_function_plot([f], [expr], points=plot_points)
    
    # Decide whether to ask for minima, maxima, or both
    problem_types = ['minima', 'maxima', 'extrema']
    problem_type = random.choice(problem_types)
    
    # For selecting a specific extremum when multiple exist
    selection_types = ['leftmost', 'rightmost', 'largest', 'smallest', 'average']
    
    if problem_type == 'minima':
        minima = [(x, y) for x, y, type_ in extrema if type_ == 'min']
        
        if not minima:
            # Fallback if no minima
            answer_int = 0
            question = f"How many local minima does the function f(x) = {expr} have in the interval [-10, 10]?"
        elif len(minima) == 1:
            # Only one minimum
            x, _ = minima[0]
            answer_int = round(x)
            question = f"Find the x-coordinate of the local minimum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
        else:
            # Multiple minima, select one
            selection = random.choice(selection_types)
            if selection == 'leftmost':
                x, _ = min(minima, key=lambda p: p[0])
                answer_int = round(x)
                question = f"Find the x-coordinate of the leftmost local minimum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'rightmost':
                x, _ = max(minima, key=lambda p: p[0])
                answer_int = round(x)
                question = f"Find the x-coordinate of the rightmost local minimum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'largest':
                x, y = max(minima, key=lambda p: p[1])
                answer_int = round(x)
                question = f"Find the x-coordinate of the local minimum with the largest y-value of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'smallest':
                x, y = min(minima, key=lambda p: p[1])
                answer_int = round(x)
                question = f"Find the x-coordinate of the local minimum with the smallest y-value of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            else:  # average
                avg_x = sum(x for x, _ in minima) / len(minima)
                answer_int = round(avg_x)
                question = f"Find the average of all x-coordinates of local minima of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
    
    elif problem_type == 'maxima':
        maxima = [(x, y) for x, y, type_ in extrema if type_ == 'max']
        
        if not maxima:
            # Fallback if no maxima
            answer_int = 0
            question = f"How many local maxima does the function f(x) = {expr} have in the interval [-10, 10]?"
        elif len(maxima) == 1:
            # Only one maximum
            x, _ = maxima[0]
            answer_int = round(x)
            question = f"Find the x-coordinate of the local maximum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
        else:
            # Multiple maxima, select one
            selection = random.choice(selection_types)
            if selection == 'leftmost':
                x, _ = min(maxima, key=lambda p: p[0])
                answer_int = round(x)
                question = f"Find the x-coordinate of the leftmost local maximum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'rightmost':
                x, _ = max(maxima, key=lambda p: p[0])
                answer_int = round(x)
                question = f"Find the x-coordinate of the rightmost local maximum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'largest':
                x, y = max(maxima, key=lambda p: p[1])
                answer_int = round(x)
                question = f"Find the x-coordinate of the local maximum with the largest y-value of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'smallest':
                x, y = min(maxima, key=lambda p: p[1])
                answer_int = round(x)
                question = f"Find the x-coordinate of the local maximum with the smallest y-value of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            else:  # average
                avg_x = sum(x for x, _ in maxima) / len(maxima)
                answer_int = round(avg_x)
                question = f"Find the average of all x-coordinates of local maxima of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
    
    else:  # extrema
        if not extrema:
            # Fallback if no extrema
            answer_int = 0
            question = f"How many local extrema does the function f(x) = {expr} have in the interval [-10, 10]?"
        elif len(extrema) == 1:
            # Only one extremum
            x, _, _ = extrema[0]
            answer_int = round(x)
            question = f"Find the x-coordinate of the local extremum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
        else:
            # Count or select specific extrema
            selection = random.choice(selection_types + ['count'])
            if selection == 'count':
                answer_int = len(extrema)
                question = f"How many local extrema (both minima and maxima) does the function f(x) = {expr} have in the interval [-10, 10]?"
            elif selection == 'leftmost':
                x, _, _ = min(extrema, key=lambda p: p[0])
                answer_int = round(x)
                question = f"Find the x-coordinate of the leftmost local extremum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'rightmost':
                x, _, _ = max(extrema, key=lambda p: p[0])
                answer_int = round(x)
                question = f"Find the x-coordinate of the rightmost local extremum of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'largest':
                # For extrema, largest by y-value
                x, y, _ = max(extrema, key=lambda p: p[1])
                answer_int = round(x)
                question = f"Find the x-coordinate of the local extremum with the largest y-value of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            elif selection == 'smallest':
                # For extrema, smallest by y-value
                x, y, _ = min(extrema, key=lambda p: p[1])
                answer_int = round(x)
                question = f"Find the x-coordinate of the local extremum with the smallest y-value of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
            else:  # average
                avg_x = sum(x for x, _, _ in extrema) / len(extrema)
                answer_int = round(avg_x)
                question = f"Find the average of all x-coordinates of local extrema of the function f(x) = {expr} in the interval [-10, 10]. Round to the nearest integer."
    
    # Convert the answer to string
    answer_str = str(answer_int)
    
    # Calculate difficulty based on function complexity and finding exact coordinates
    base_difficulty = f_info.get('difficulty', 3)
    
    # Finding coordinates is harder than counting
    coordinate_factor = 1
    
    # More extrema points increase difficulty
    extrema_factor = min(1, len(extrema) - 1) if len(extrema) > 1 else 0
    
    # Final difficulty
    final_difficulty = base_difficulty + coordinate_factor + extrema_factor

    difficulty_info = {
        "level": final_difficulty,
        "functions": [f_info],
        "num_extrema": len(extrema),
        "extrema": extrema,
        "problem_type": problem_type
    }
    
    return Problem(question=question, answer=answer_str, plot_data=plot_data, difficulty=difficulty_info)

def generate_area_problem(composition_nums=2):
    """
    Generate a problem about finding the area between two curves
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate functions with specified composition level, if provided
    f1, expr1, f1_info = generate_composite_function(composition_nums)
    f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))

    # Find intersections to determine integration bounds using a while loop
    intersections = find_intersections(f1, f2, x_min=-5, x_max=5)
    
    # Use a while loop to keep trying until we find at least 2 intersections
    max_attempts = 10
    attempts = 0
    
    while len(intersections) < 2 and attempts < max_attempts:
        attempts += 1
        # Try with different functions, starting with simpler ones for easier integration
        f1, expr1, f1_info = generate_composite_function(composition_nums)
        f2, expr2, f2_info = generate_composite_function(max(1, composition_nums-1))
        
        intersections = find_intersections(f1, f2, x_min=-5, x_max=5)
    
    # Select two intersections to form a bounded region
    if len(intersections) >= 2:
        # Sort and pick consecutive intersections
        intersections.sort()
        idx = random.randint(0, len(intersections) - 2)
        x_left, x_right = intersections[idx], intersections[idx + 1]
    else:
        # If still no luck, use fixed interval
        x_left, x_right = -3, 3
    
    # Round bounds to 1 decimal place
    x_left = round(x_left, 1)
    x_right = round(x_right, 1)

    # Ensure x_left and x_right are different after rounding
    if x_left == x_right:
        # Adjust x_right to be at least 0.1 greater than x_left
        x_right = x_left + 0.1
        # If the adjusted bound is outside our search range, use fixed values
        if x_right > 5:
            x_left, x_right = -3, 3
    
    # Compute the area between the curves
    try:
        def area_function(x):
            return abs(f1(x) - f2(x))
        
        area, _ = integrate.quad(area_function, x_left, x_right)
        # Round to 1 decimal place
        area = round(area, 1)
    except:
        # Fallback if integration fails
        area = 5.0  # Just a placeholder
    
    # Create plot points for the intersections
    plot_points = [
        (x_left, f1(x_left), f"A ({x_left}, {round(f1(x_left), 2)})"),
        (x_right, f1(x_right), f"B ({x_right}, {round(f1(x_right), 2)})")
    ]
    
    # Create plot data
    plot_data = create_function_plot([f1, f2], [expr1, expr2], points=plot_points)
    
    templates = [
        f"Find the area of the region bounded by the curves f(x) = {expr1}, g(x) = {expr2}, x = {x_left}, and x = {x_right}. Round your answer to 1 decimal place.",
        f"Calculate the area enclosed between the functions f(x) = {expr1} and g(x) = {expr2} from x = {x_left} to x = {x_right}. Round to 1 decimal place.",
        f"Determine the area of the region enclosed by the graphs of f(x) = {expr1} and g(x) = {expr2} between the vertical lines x = {x_left} and x = {x_right}. Round to 1 decimal place."
    ]
    
    question = random.choice(templates)
    
    # Calculate difficulty based on function types and area calculation
    base_difficulty = f1_info.get('difficulty', 2) + f2_info.get('difficulty', 2)
    
    # Area calculation is naturally more difficult
    area_factor = 2
    
    # Final difficulty
    final_difficulty = base_difficulty + area_factor
    
    difficulty_info = {
        "level": final_difficulty,
        "functions": [f1_info, f2_info],
        "bounds": [x_left, x_right]
    }
    
    return Problem(question=question, answer=area, plot_data=plot_data, difficulty=difficulty_info)

def generate_zero_problem(composition_nums=None):
    """
    Generate a problem about finding zeros of a function
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate a function with the specified number of compositions
    if composition_nums is not None:
        f, expr, f_info = generate_composite_function(composition_nums)
    else:
        # Generate a random function if composition_nums is None
        f, expr, f_info = generate_random_function()
    
    # Find zeros by looking for intersections with y = 0
    def zero_func(x):
        return 0
    
    zeros = find_intersections(f, zero_func)
    
    while not zeros:
        if composition_nums is not None:
            f, expr, f_info = generate_composite_function(composition_nums)
        else:
            f, expr, f_info = generate_random_function()
        zeros = find_intersections(f, zero_func)
    
    # Create plot points for zeros
    plot_points = [(x, 0, f"zero ({x}, 0)") for x in zeros]
    
    # Create plot data
    plot_data = create_function_plot([f], [expr], points=plot_points)
    
    templates = [
        f"How many zeros does the function f(x) = {expr} have in the interval [-10, 10]?",
        f"Find the number of x-intercepts of the function f(x) = {expr} where -10 ≤ x ≤ 10.",
        f"How many solutions does the equation {expr} = 0 have in the range [-10, 10]?",
        f"Determine the number of values x in [-10, 10] where f(x) = {expr} equals zero."
    ]
    
    question = random.choice(templates)
    answer = len(zeros)
    
    # Calculate difficulty based on function complexity and number of zeros
    base_difficulty = f_info.get('difficulty', 2)
    
    # More zeros can indicate higher difficulty
    zero_factor = int(len(zeros) ** 0.75)
    
    # Final difficulty
    final_difficulty = base_difficulty + zero_factor
    
    difficulty_info = {
        "level": final_difficulty,
        "functions": [f_info],
        "num_zeros": len(zeros)
    }
    
    return Problem(question=question, answer=answer, plot_data=plot_data, difficulty=difficulty_info)

def generate_derivative_sign_problem(composition_nums=None):
    """
    Generate a problem about determining where the derivative is positive/negative
    
    Args:
        composition_nums: Integer specifying the number of functions to compose
                         (None for a non-composite function)
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    # Generate a function with the specified number of compositions
    if composition_nums is not None:
        f, expr, f_info = generate_composite_function(composition_nums)
    else:
        # Generate a random function if composition_nums is None
        f, expr, f_info = generate_random_function()
    
    # Define the derivative (numerically)
    def f_prime(x):
        h = 1e-5
        try:
            # Calculate function values at neighborhood points
            f_plus = f(x + h)
            f_minus = f(x - h)
            
            # Check for invalid values
            if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                return 0  # Return 0 for invalid values
            
            return (f_plus - f_minus) / (2 * h)
        except:
            return 0  # Return 0 on error
    
    # Sample points to find where derivative is positive/negative
    x_samples = np.linspace(-10, 10, 200)
    y_prime_samples = []
    
    for x in x_samples:
        try:
            y_prime = f_prime(x)
            if np.isfinite(y_prime) and abs(y_prime) < 1000:
                y_prime_samples.append((x, y_prime))
        except:
            continue
    
    if not y_prime_samples:
        # If no valid derivatives, try a different function
        f, expr, f_info = generate_quadratic_function()
        y_prime_samples = []
        for x in x_samples:
            try:
                y_prime = f_prime(x)
                if np.isfinite(y_prime) and abs(y_prime) < 1000:
                    y_prime_samples.append((x, y_prime))
            except:
                continue
    
    # Count intervals where derivative is positive
    positive_intervals = []
    negative_intervals = []
    
    if y_prime_samples:
        x_vals, y_primes = zip(*y_prime_samples)
        sign_changes = []
        
        # Find sign changes
        for i in range(1, len(y_primes)):
            if y_primes[i-1] * y_primes[i] <= 0 and abs(y_primes[i-1]) > 1e-6 and abs(y_primes[i]) > 1e-6:
                # Linear interpolation to find the x-value where sign changes
                x0, y0 = x_vals[i-1], y_primes[i-1]
                x1, y1 = x_vals[i], y_primes[i]
                
                if abs(y1 - y0) > 1e-6:
                    x_intercept = x0 - y0 * (x1 - x0) / (y1 - y0)
                    sign_changes.append(round(x_intercept, 2))
        
        # Create intervals
        intervals = [-10] + sign_changes + [10]
        
        # Determine sign in each interval
        for i in range(len(intervals) - 1):
            start, end = intervals[i], intervals[i+1]
            mid = (start + end) / 2
            
            try:
                if f_prime(mid) > 0:
                    positive_intervals.append((start, end))
                else:
                    negative_intervals.append((start, end))
            except:
                # Skip problematic intervals
                continue
    
    # Create plot data
    def derivative_func(x):
        return f_prime(x)
    
    plot_data = create_function_plot(
        [f, derivative_func], 
        [expr, f"f'(x) - derivative of {expr}"]
    )
    
    # Choose which type of problem to create
    problem_types = ['count_positive', 'count_negative']
    problem_type = random.choice(problem_types)
    
    if problem_type == 'count_positive':
        templates = [
            f"On how many maximal connected intervals within [-10, 10] is the derivative of f(x) = {expr} positive?",
            f"For the function f(x) = {expr}, find the number of maximal connected intervals in [-10, 10] where f'(x) > 0.",
            f"Determine the number of maximal connected intervals in [-10, 10] where the function f(x) = {expr} is strictly increasing."
        ]
        answer = len(positive_intervals)
    else:  # count_negative
        templates = [
            f"On how many maximal connected intervals within [-10, 10] is the derivative of f(x) = {expr} negative?",
            f"For the function f(x) = {expr}, find the number of maximal connected intervals in [-10, 10] where f'(x) < 0.",
            f"Determine the number of maximal connected intervals in [-10, 10] where the function f(x) = {expr} is strictly decreasing."
        ]
        answer = len(negative_intervals)
    
    question = random.choice(templates)
    
    # Calculate difficulty based on function complexity and analysis of derivative
    base_difficulty = f_info.get('difficulty', 3)
    
    derivative_factor = 2
    
    interval_count = int(max(len(positive_intervals) ** 0.75, len(negative_intervals) ** 0.75))

    final_difficulty = base_difficulty + derivative_factor + interval_count
    
    difficulty_info = {
        "level": final_difficulty,
        "functions": [f_info],
        "positive_intervals": len(positive_intervals),
        "negative_intervals": len(negative_intervals),
        "problem_type": problem_type
    }
    
    return Problem(question=question, answer=answer, plot_data=plot_data, difficulty=difficulty_info)

# --- Main Function and CLI Interface ---

def main():
    """
    Command-line interface for generating function analysis problems.
    """
    parser = argparse.ArgumentParser(description='Generate function analysis problems.')
    parser.add_argument('--type', type=str,
                      choices=['intersection', 'intersection_coords', 'extrema', 
                               'extrema_coords', 'area', 'zeros', 'derivative_sign', 'all'], 
                      default='intersection', help='Type of function problem to generate')
    parser.add_argument('--count', type=int, default=10, help='Number of problems to generate')
    parser.add_argument('--difficulty', type=int, choices=[1, 2, 3, 4, 5], default=None, 
                     help='Difficulty level (1-5, where 5 is hardest)')
    parser.add_argument('--composition', type=int, choices=[1, 2, 3, 4], default=2,
                     help='Number of functions to compose (1-4, where higher values create more complex functions)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--save_plots', default=True, help='Save plots to files')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.save_plots:
        import os
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

    if args.type == 'all':
        problem_types = ['intersection', 'intersection_coords', 'extrema', 
                         'extrema_coords', 'area', 'zeros', 'derivative_sign']
    else:
        problem_types = [args.type]

    for _ in range(args.count):
        for problem_type in problem_types:
            print(f"\n--- {problem_type.upper()} PROBLEM ---")
            
            if problem_type == 'intersection':
                problem = generate_intersection_problem(args.composition)
            elif problem_type == 'intersection_coords':
                problem = generate_intersection_coordinates_problem(args.composition)
            elif problem_type == 'extrema':
                problem = generate_extrema_problem(args.composition)
            elif problem_type == 'extrema_coords':
                problem = generate_extrema_coordinates_problem(args.composition)
            elif problem_type == 'area':
                problem = generate_area_problem(args.composition)
            elif problem_type == 'zeros':
                problem = generate_zero_problem(args.composition)
            elif problem_type == 'derivative_sign':
                problem = generate_derivative_sign_problem(args.composition)
            
            print(f"Question: {problem.question}")
            print(f"\nAnswer: {problem.answer}")
            
            # Print difficulty information if available
            if hasattr(problem, 'difficulty') and problem.difficulty:
                print(f"\nDifficulty Level: {problem.difficulty.get('level', 'Unknown')}")
                
                # Print function types used in the problem
                functions = problem.difficulty.get('functions', [])
                if functions:
                    print("Functions used:")
                    for i, func_info in enumerate(functions):
                        func_type = func_info.get('type', 'unknown')
                        if func_type == 'composite':
                            components = func_info.get('component_types', [])
                            print(f"  Function {i+1}: Composite of {', '.join(components)}")
                        else:
                            print(f"  Function {i+1}: {func_type}")
            
            if args.save_plots and problem.plot_data:
                # Save the plot
                plot_file = f"{args.plot_dir}/{problem_type}_{_}.png"
                with open(plot_file, 'wb') as f:
                    f.write(base64.b64decode(problem.plot_data))
                print(f"Plot saved to {plot_file}")
            
            print("-" * 50)


if __name__ == "__main__":
    main() 