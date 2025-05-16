"""
Function debugging module to help analyze and visualize function descriptions from text.
This module helps in debugging and verifying problems related to function analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import sympy as sp
from io import BytesIO
import base64
import sys
import os

# Add the parent directory to sys.path to import from modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.function_analysis import find_intersections, create_function_plot

def parse_function_from_text(func_text):
    """
    Parse a function expression from text and convert it to a callable function.
    
    Args:
        func_text: String description of the function
        
    Returns:
        A callable function that implements the described function
    """
    # Replace all instances of 'e^' with 'np.exp'
    func_text = func_text.replace('e^', 'np.exp')
    
    # Replace all instances of sin, cos with np.sin, np.cos
    func_text = func_text.replace('sin', 'np.sin')
    func_text = func_text.replace('cos', 'np.cos')
    
    # Replace all instances of ln with np.log
    func_text = func_text.replace('ln', 'np.log')
    
    # Replace ^ with ** for exponentiation
    func_text = func_text.replace('^', '**')
    
    # Replace π or pi with np.pi
    func_text = func_text.replace('π', 'np.pi')
    func_text = func_text.replace(' pi ', ' np.pi ')
    
    # Replace |x| with abs(x)
    # Use regex to find patterns like |expression| and replace with abs(expression)
    func_text = re.sub(r'\|([^|]+)\|', r'abs(\1)', func_text)
    
    # Create a lambda function
    try:
        f = lambda x: eval(func_text)
        # Test the function with a sample value to catch errors early
        f(0)
        return f
    except Exception as e:
        print(f"Error parsing function: {e}")
        print(f"Function text: {func_text}")
        return None

def extract_functions_from_question(question_text):
    """
    Extract function descriptions from a problem question.
    
    Args:
        question_text: The text of the problem
        
    Returns:
        A tuple of (f, g) functions and their string expressions
    """
    # Common patterns for functions in questions
    patterns = [
        r'f\(x\)\s*=\s*([^,]+) and g\(x\)\s*=\s*([^,\.]+)',
        r'functions f\(x\)\s*=\s*([^,]+) and g\(x\)\s*=\s*([^,\.]+)',
        r'the functions f\(x\)\s*=\s*([^,]+) and g\(x\)\s*=\s*([^,\.]+)',
        r'graphs of f\(x\)\s*=\s*([^,]+) and g\(x\)\s*=\s*([^,\.]+)',
        r'graphs of ([^,]+) and ([^,\.]+) intersect',
        r'equation ([^=]+) = ([^,\.]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question_text)
        if match:
            f_expr = match.group(1).strip()
            g_expr = match.group(2).strip()
            
            # Parse the functions
            f = parse_function_from_text(f_expr)
            g = parse_function_from_text(g_expr)
            
            return (f, f_expr), (g, g_expr)
    
    print("Could not extract functions from the question.")
    return None, None

def debug_function_problem(question_text, x_min=-10, x_max=10):
    """
    Debug a function analysis problem by extracting functions, 
    finding intersections, and creating visualizations.
    
    Args:
        question_text: The text of the problem
        x_min: Minimum x value for analysis (default: -10)
        x_max: Maximum x value for analysis (default: 10)
        
    Returns:
        A dictionary with analysis results
    """
    print(f"Analyzing problem: {question_text}")
    
    # Extract functions from the question
    (f, f_expr), (g, g_expr) = extract_functions_from_question(question_text)
    
    if f is None or g is None:
        return {"error": "Could not parse functions from the question text"}
    
    print(f"Function f(x) = {f_expr}")
    print(f"Function g(x) = {g_expr}")
    
    # Find intersections
    intersections = find_intersections(f, g, x_min=x_min, x_max=x_max)
    print(f"Found {len(intersections)} intersection points: {intersections}")
    
    # Create a plot
    plot_points = [(x, f(x), f"({x}, {round(f(x), 2)})") for x in intersections]
    plot_data = create_function_plot([f, g], [f_expr, g_expr], points=plot_points)
    
    # Save the plot to a file
    img_data = base64.b64decode(plot_data)
    plot_filename = "debug_plot.png"
    with open(plot_filename, "wb") as file:
        file.write(img_data)
    print(f"Plot saved to {plot_filename}")
    
    # Sample some points for verification
    x_sample = np.linspace(x_min, x_max, 10)
    f_values = []
    g_values = []
    
    print("\nSample points for verification:")
    print("x\t\tf(x)\t\tg(x)\t\tf(x)-g(x)")
    print("-" * 50)
    
    for x in x_sample:
        try:
            f_val = f(x)
            g_val = g(x)
            if np.isfinite(f_val) and np.isfinite(g_val):
                f_values.append((x, f_val))
                g_values.append((x, g_val))
                print(f"{x:.2f}\t\t{f_val:.4f}\t\t{g_val:.4f}\t\t{f_val-g_val:.4f}")
        except Exception as e:
            print(f"Error at x={x}: {e}")
    
    return {
        "f_expr": f_expr,
        "g_expr": g_expr,
        "intersections": intersections,
        "plot_file": plot_filename,
        "f_values": f_values,
        "g_values": g_values
    }

if __name__ == "__main__":
    # Example usage
    question = "Determine the number of intersection points between the functions f(x) = -1 and g(x) = -2*sin(3*π*x + 1) + 1"
    
    result = debug_function_problem(question)
    print("\nAnalysis result:")
    print(f"Number of intersections: {len(result['intersections'])}")
    print(f"Intersection points: {result['intersections']}")
    print(f"Plot saved to: {result['plot_file']}")

