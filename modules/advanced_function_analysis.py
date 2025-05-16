"""
Enhanced function analysis module supporting various function representations.
This module generates problems related to functions in forms like y = f(x), x = g(y),
and implicit functions f(x,y) = 0.
"""

import numpy as np
import random
import argparse
import math
from collections import namedtuple
from scipy import optimize, integrate
import sympy as sp
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Simple Problem container
Problem = namedtuple('Problem', ('question', 'answer', 'plot_data'))

# Function type enumeration
class FunctionType:
    Y_OF_X = "y_of_x"         # y = f(x)
    X_OF_Y = "x_of_y"         # x = f(y)
    IMPLICIT = "implicit"     # f(x,y) = 0
    PARAMETRIC = "parametric" # x = f(t), y = g(t)

# --- Function Representation Classes ---

class Function:
    """Base class for all function representations"""
    def __init__(self, func_type, expr_str, domain=(-10, 10)):
        self.func_type = func_type
        self.expr_str = expr_str
        self.domain = domain
    
    def __str__(self):
        return self.expr_str
    
    def evaluate(self, *args):
        """Evaluate the function - to be implemented by subclasses"""
        raise NotImplementedError
        
    def get_plot_data(self, num_points=1000):
        """Return data for plotting the function - to be implemented by subclasses"""
        raise NotImplementedError

class FunctionYofX(Function):
    """Function of the form y = f(x)"""
    def __init__(self, func, expr_str, domain=(-10, 10)):
        super().__init__(FunctionType.Y_OF_X, expr_str, domain)
        self.func = func
    
    def evaluate(self, x):
        return self.func(x)
    
    def get_plot_data(self, num_points=1000):
        x_vals = np.linspace(self.domain[0], self.domain[1], num_points)
        y_vals = []
        valid_x = []
        
        for x in x_vals:
            try:
                y = self.evaluate(x)
                if np.isfinite(y) and -1000 < y < 1000:  # Filter extreme values
                    valid_x.append(x)
                    y_vals.append(y)
            except:
                continue
                
        return valid_x, y_vals

class FunctionXofY(Function):
    """Function of the form x = f(y)"""
    def __init__(self, func, expr_str, domain=(-10, 10)):
        super().__init__(FunctionType.X_OF_Y, expr_str, domain)
        self.func = func
    
    def evaluate(self, y):
        return self.func(y)
    
    def get_plot_data(self, num_points=1000):
        y_vals = np.linspace(self.domain[0], self.domain[1], num_points)
        x_vals = []
        valid_y = []
        
        for y in y_vals:
            try:
                x = self.evaluate(y)
                if np.isfinite(x) and -1000 < x < 1000:  # Filter extreme values
                    x_vals.append(x)
                    valid_y.append(y)
            except:
                continue
                
        return x_vals, valid_y

class ImplicitFunction(Function):
    """Implicit function of the form f(x,y) = 0"""
    def __init__(self, func, expr_str, x_domain=(-10, 10), y_domain=(-10, 10)):
        super().__init__(FunctionType.IMPLICIT, expr_str, x_domain)
        self.func = func
        self.y_domain = y_domain
    
    def evaluate(self, x, y):
        return self.func(x, y)
    
    def get_plot_data(self, num_points=100):
        # For implicit functions, we'll use a contour plot approach
        x = np.linspace(self.domain[0], self.domain[1], num_points)
        y = np.linspace(self.y_domain[0], self.y_domain[1], num_points)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = self.evaluate(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan
        
        return X, Y, Z

class ParametricFunction(Function):
    """Parametric function of the form x = f(t), y = g(t)"""
    def __init__(self, func_x, func_y, expr_x_str, expr_y_str, t_domain=(0, 2*np.pi)):
        expr_str = f"x = {expr_x_str}, y = {expr_y_str}"
        super().__init__(FunctionType.PARAMETRIC, expr_str, t_domain)
        self.func_x = func_x
        self.func_y = func_y
        self.expr_x_str = expr_x_str
        self.expr_y_str = expr_y_str
    
    def evaluate(self, t):
        return self.func_x(t), self.func_y(t)
    
    def get_plot_data(self, num_points=1000):
        t_vals = np.linspace(self.domain[0], self.domain[1], num_points)
        x_vals = []
        y_vals = []
        
        for t in t_vals:
            try:
                x, y = self.evaluate(t)
                if (np.isfinite(x) and np.isfinite(y) and 
                    -1000 < x < 1000 and -1000 < y < 1000):
                    x_vals.append(x)
                    y_vals.append(y)
            except:
                continue
                
        return x_vals, y_vals

# --- Function Generation Utilities ---

def generate_linear_function(min_coef=-5, max_coef=5, func_type=FunctionType.Y_OF_X):
    """Generate a linear function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:  # Avoid horizontal/vertical lines
        a = random.choice([-1, 1])
    b = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
        def f(x):
            return a * x + b
        
        expr = f"{a}x"
        if b > 0:
            expr += f" + {b}"
        elif b < 0:
            expr += f" - {abs(b)}"
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            return (y - b) / a if a != 0 else np.nan
        
        expr = f"(y"
        if b > 0:
            expr += f" - {b}"
        elif b < 0:
            expr += f" + {abs(b)}"
        expr += f") / {a}" if a != 1 else expr + ")"
            
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_quadratic_function(min_coef=-5, max_coef=5, func_type=FunctionType.Y_OF_X):
    """Generate a quadratic function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:  # Avoid linear functions
        a = random.choice([-1, 1])
    b = random.randint(min_coef, max_coef)
    c = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
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
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # x = (-b ± sqrt(b^2 - 4a(c-y))) / 2a
            # We'll return the positive solution for simplicity
            discriminant = b**2 - 4*a*(c-y)
            if discriminant < 0:
                return np.nan
            return (-b + np.sqrt(discriminant)) / (2*a)
        
        expr = f"(-{b} + sqrt({b}^2 - 4·{a}·({c}-y))) / (2·{a})"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_sin_function(min_coef=-3, max_coef=3, min_freq=1, max_freq=3, func_type=FunctionType.Y_OF_X):
    """Generate a sine function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(min_freq, max_freq)
    c = random.randint(-3, 3)
    d = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
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
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # Solve for x in y = a * sin(b * pi * x + c) + d
            # This is y - d = a * sin(b * pi * x + c)
            # For simplicity, we'll just return one possible value
            # using arcsin, which is valid for -1 <= (y-d)/a <= 1
            try:
                arg = (y - d) / a
                if abs(arg) > 1:
                    return np.nan
                return (np.arcsin(arg) - c) / (b * np.pi)
            except:
                return np.nan
        
        expr = f"(arcsin((y - {d})/{a}) - {c}) / ({b}π)"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_cos_function(min_coef=-3, max_coef=3, min_freq=1, max_freq=3, func_type=FunctionType.Y_OF_X):
    """Generate a cosine function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(min_freq, max_freq)
    c = random.randint(-3, 3)
    d = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
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
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # Similar to sine case but with arccos
            try:
                arg = (y - d) / a
                if abs(arg) > 1:
                    return np.nan
                return (np.arccos(arg) - c) / (b * np.pi)
            except:
                return np.nan
        
        expr = f"(arccos((y - {d})/{a}) - {c}) / ({b}π)"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_abs_function(min_coef=-3, max_coef=3, func_type=FunctionType.Y_OF_X):
    """Generate an absolute value function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(-3, 3)
    c = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
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
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # Solve for x in y = a*|x-b| + c
            # (y-c)/a = |x-b|
            # x = b ± (y-c)/a
            # We'll return both solutions when valid
            arg = (y - c) / a
            if arg < 0:  # No solution when |x-b| would be negative
                return np.nan
            # Return the positive solution for simplicity
            return b + arg
        
        expr = f"{b} + (y - {c})/{a}"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_exp_function(min_coef=-3, max_coef=3, func_type=FunctionType.Y_OF_X):
    """Generate an exponential function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(-2, 2)
    if b == 0:
        b = random.choice([-1, 1])
    c = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
        def f(x):
            return a * np.exp(b * x) + c
        
        expr = f"{a}e^({b}x)"
        if c > 0:
            expr += f" + {c}"
        elif c < 0:
            expr += f" - {abs(c)}"
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # Solve for x in y = a*e^(b*x) + c
            # (y-c)/a = e^(b*x)
            # ln((y-c)/a) = b*x
            # x = ln((y-c)/a) / b
            try:
                return np.log((y - c) / a) / b
            except:
                return np.nan
        
        expr = f"ln((y - {c})/{a}) / {b}"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_log_function(min_coef=-3, max_coef=3, func_type=FunctionType.Y_OF_X):
    """Generate a logarithmic function with integer coefficients"""
    a = random.randint(min_coef, max_coef)
    if a == 0:
        a = random.choice([-2, -1, 1, 2])
    b = random.randint(1, 3)  # Positive to ensure domain validity
    c = random.randint(1, 5)  # Ensure bx + c > 0 for x ≥ 0
    d = random.randint(min_coef, max_coef)
    
    if func_type == FunctionType.Y_OF_X:
        def f(x):
            # Handle domain issues with a small epsilon to prevent errors
            if b * x + c <= 0:
                return np.nan
            return a * np.log(b * x + c) + d
        
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
            
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # Solve for x in y = a*ln(b*x + c) + d
            # (y-d)/a = ln(b*x + c)
            # e^((y-d)/a) = b*x + c
            # x = (e^((y-d)/a) - c) / b
            try:
                return (np.exp((y - d) / a) - c) / b
            except:
                return np.nan
        
        expr = f"(e^((y - {d})/{a}) - {c}) / {b}"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_rational_function(min_coef=-3, max_coef=3, func_type=FunctionType.Y_OF_X):
    """Generate a rational function with integer coefficients"""
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
    
    if func_type == FunctionType.Y_OF_X:
        def f(x):
            # Check for division by zero
            denom = c * x + d
            if abs(denom) < 1e-10:
                return np.nan
            return (a * x + b) / denom
        
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
        return FunctionYofX(f, expr)
    
    elif func_type == FunctionType.X_OF_Y:
        def f(y):
            # Solve for x in y = (ax+b)/(cx+d)
            # y(cx+d) = ax+b
            # ycx - ax = b - yd
            # x(yc-a) = b-yd
            # x = (b-yd)/(yc-a)
            try:
                denom = y * c - a
                if abs(denom) < 1e-10:
                    return np.nan
                return (b - y * d) / denom
            except:
                return np.nan
        
        expr = f"({b} - y·{d})/(y·{c} - {a})"
        return FunctionXofY(f, expr)
    
    else:
        raise ValueError(f"Unsupported function type: {func_type}")

def generate_circle_function(radius=None, center=None):
    """Generate a circle as an implicit function"""
    if radius is None:
        radius = random.randint(1, 5)
    if center is None:
        h = random.randint(-3, 3)
        k = random.randint(-3, 3)
    else:
        h, k = center
    
    def f(x, y):
        return (x - h)**2 + (y - k)**2 - radius**2
    
    expr = f"(x - {h})^2 + (y - {k})^2 = {radius}^2"
    
    return ImplicitFunction(f, expr)

def generate_ellipse_function(a=None, b=None, center=None):
    """Generate an ellipse as an implicit function"""
    if a is None:
        a = random.randint(2, 6)
    if b is None:
        b = random.randint(2, 6)
    if center is None:
        h = random.randint(-3, 3)
        k = random.randint(-3, 3)
    else:
        h, k = center
    
    def f(x, y):
        return ((x - h)/a)**2 + ((y - k)/b)**2 - 1
    
    expr = f"((x - {h})/{a})^2 + ((y - {k})/{b})^2 = 1"
    
    return ImplicitFunction(f, expr)

def generate_hyperbola_function():
    """Generate a hyperbola as an implicit function"""
    a = random.randint(2, 5)
    b = random.randint(2, 5)
    h = random.randint(-3, 3)
    k = random.randint(-3, 3)
    
    # Randomly choose between horizontal and vertical orientation
    orientation = random.choice(['horizontal', 'vertical'])
    
    if orientation == 'horizontal':
        def f(x, y):
            return ((x - h)/a)**2 - ((y - k)/b)**2 - 1
        
        expr = f"((x - {h})/{a})^2 - ((y - {k})/{b})^2 = 1"
    else:
        def f(x, y):
            return ((y - k)/a)**2 - ((x - h)/b)**2 - 1
        
        expr = f"((y - {k})/{a})^2 - ((x - {h})/{b})^2 = 1"
    
    return ImplicitFunction(f, expr)

def generate_parabola_implicit_function():
    """Generate a parabola as an implicit function"""
    # Focus point and directrix parameters
    p = random.randint(1, 3)  # Distance from vertex to focus
    h = random.randint(-3, 3)  # x-coordinate of vertex
    k = random.randint(-3, 3)  # y-coordinate of vertex
    
    # Randomly choose between vertical and horizontal orientation
    orientation = random.choice(['vertical', 'horizontal'])
    
    if orientation == 'vertical':
        def f(x, y):
            # Standard form: (x - h)^2 = 4p(y - k)
            return (x - h)**2 - 4*p*(y - k)
        
        expr = f"(x - {h})^2 = {4*p}(y - {k})"
    else:
        def f(x, y):
            # Standard form: (y - k)^2 = 4p(x - h)
            return (y - k)**2 - 4*p*(x - h)
        
        expr = f"(y - {k})^2 = {4*p}(x - {h})"
    
    return ImplicitFunction(f, expr)

def generate_parametric_circle(radius=None, center=None):
    """Generate a circle as a parametric function"""
    if radius is None:
        radius = random.randint(1, 5)
    if center is None:
        h = random.randint(-3, 3)
        k = random.randint(-3, 3)
    else:
        h, k = center
    
    def f_x(t):
        return h + radius * np.cos(t)
    
    def f_y(t):
        return k + radius * np.sin(t)
    
    expr_x = f"{h} + {radius}cos(t)"
    expr_y = f"{k} + {radius}sin(t)"
    
    return ParametricFunction(f_x, f_y, expr_x, expr_y)

def generate_parametric_ellipse(a=None, b=None, center=None):
    """Generate an ellipse as a parametric function"""
    if a is None:
        a = random.randint(2, 6)
    if b is None:
        b = random.randint(2, 6)
    if center is None:
        h = random.randint(-3, 3)
        k = random.randint(-3, 3)
    else:
        h, k = center
    
    def f_x(t):
        return h + a * np.cos(t)
    
    def f_y(t):
        return k + b * np.sin(t)
    
    expr_x = f"{h} + {a}cos(t)"
    expr_y = f"{k} + {b}sin(t)"
    
    return ParametricFunction(f_x, f_y, expr_x, expr_y)

def generate_parametric_lissajous():
    """Generate a Lissajous curve as a parametric function"""
    a = random.randint(2, 5)
    b = random.randint(2, 5)
    delta = random.uniform(0, 2*np.pi)  # Phase difference
    
    def f_x(t):
        return a * np.sin(b*t + delta)
    
    def f_y(t):
        return b * np.sin(t)
    
    expr_x = f"{a}sin({b}t + {delta:.2f})"
    expr_y = f"{b}sin(t)"
    
    return ParametricFunction(f_x, f_y, expr_x, expr_y, t_domain=(0, 2*np.pi))

def generate_spiral_function():
    """Generate a spiral as a parametric function"""
    a = random.uniform(0.1, 0.5)  # Controls how quickly the spiral expands
    
    def f_x(t):
        return a * t * np.cos(t)
    
    def f_y(t):
        return a * t * np.sin(t)
    
    expr_x = f"{a:.2f}·t·cos(t)"
    expr_y = f"{a:.2f}·t·sin(t)"
    
    return ParametricFunction(f_x, f_y, expr_x, expr_y, t_domain=(0, 6*np.pi))

def generate_parametric_function():
    """Generate a random parametric function"""
    choices = [
        generate_parametric_circle,
        generate_parametric_ellipse,
        generate_parametric_lissajous,
        generate_spiral_function
    ]
    
    generator = random.choice(choices)
    return generator()

def generate_composite_function_y_of_x(num_components=2):
    """
    Generate a composite function by combining simpler y=f(x) functions
    
    Args:
        num_components: Number of functions to compose (default 2)
    
    Returns:
        A FunctionYofX object representing the composite function
    """
    # Choose the number of components (2 by default)
    if num_components < 1:
        num_components = 1
    
    # Available function generators for Y_OF_X
    generators = [
        lambda: generate_linear_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_quadratic_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_sin_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_cos_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_abs_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_exp_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_log_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_rational_function(func_type=FunctionType.Y_OF_X)
    ]
    
    # Choose random generators
    chosen_generators = [random.choice(generators) for _ in range(num_components)]
    
    # Generate the component functions
    components = [gen() for gen in chosen_generators]
    
    # Unpack the components
    funcs = [comp.func for comp in components]
    exprs = [comp.expr_str for comp in components]
    
    # Create the composite function
    def composite_func(x):
        result = funcs[0](x)
        for func in funcs[1:]:
            try:
                result = func(result)
                if not np.isfinite(result):
                    return np.nan
            except:
                return np.nan
        return result
    
    # Create the expression string
    composite_expr = exprs[-1]
    for expr in reversed(exprs[:-1]):
        composite_expr = composite_expr.replace("x", f"({expr})")
    
    return FunctionYofX(composite_func, composite_expr)

def generate_random_function(allow_implicit=True, allow_parametric=True):
    """Generate a random function of any supported type"""
    func_types = [FunctionType.Y_OF_X, FunctionType.X_OF_Y]
    
    if allow_implicit:
        func_types.append(FunctionType.IMPLICIT)
    
    if allow_parametric:
        func_types.append(FunctionType.PARAMETRIC)
    
    chosen_type = random.choice(func_types)
    
    if chosen_type == FunctionType.Y_OF_X:
        generators = [
            lambda: generate_linear_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_quadratic_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_sin_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_cos_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_abs_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_exp_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_log_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_rational_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_composite_function_y_of_x()
        ]
        generator = random.choice(generators)
        return generator()
    
    elif chosen_type == FunctionType.X_OF_Y:
        generators = [
            lambda: generate_linear_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_quadratic_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_sin_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_cos_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_abs_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_exp_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_log_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_rational_function(func_type=FunctionType.X_OF_Y)
        ]
        generator = random.choice(generators)
        return generator()
    
    elif chosen_type == FunctionType.IMPLICIT:
        generators = [
            generate_circle_function,
            generate_ellipse_function,
            generate_hyperbola_function,
            generate_parabola_implicit_function
        ]
        generator = random.choice(generators)
        return generator()
    
    elif chosen_type == FunctionType.PARAMETRIC:
        return generate_parametric_function()
    
    else:
        raise ValueError(f"Unsupported function type: {chosen_type}")

# --- Analysis Utilities ---

def find_intersections_y_of_x(func1, func2, x_min=-10, x_max=10, num_points=1000):
    """Find intersection points of two y=f(x) functions"""
    # Create a function representing the difference
    def diff(x):
        return func1.evaluate(x) - func2.evaluate(x)
    
    # Sample points to look for sign changes
    x_samples = np.linspace(x_min, x_max, num_points)
    y_diff = []
    valid_x = []
    
    for x in x_samples:
        try:
            y = diff(x)
            if np.isfinite(y):
                y_diff.append(y)
                valid_x.append(x)
        except:
            continue
    
    if not valid_x:
        return []
    
    # Find sign changes
    sign_changes = []
    for i in range(1, len(valid_x)):
        if y_diff[i-1] * y_diff[i] <= 0 and abs(y_diff[i-1]) + abs(y_diff[i]) > 1e-10:
            sign_changes.append(i)
    
    # For each sign change, find the precise root
    intersections = []
    for i in sign_changes:
        x_left, x_right = valid_x[i-1], valid_x[i]
        try:
            root, = optimize.fsolve(diff, (x_left + x_right) / 2)
            # Check if the root is valid and within bounds
            if x_min <= root <= x_max and abs(diff(root)) < 1e-5:
                # Round to 2 decimal places
                intersections.append(round(root, 2))
        except:
            continue
    
    # Remove duplicates and sort
    intersections = sorted(list(set(intersections)))
    return intersections

def find_intersections(func1, func2):
    """
    Find intersections between two functions of any supported type
    
    Returns:
        List of (x, y) coordinates of intersection points
    """
    # Handle different function type combinations
    if func1.func_type == FunctionType.Y_OF_X and func2.func_type == FunctionType.Y_OF_X:
        x_values = find_intersections_y_of_x(func1, func2)
        return [(x, func1.evaluate(x)) for x in x_values]
    
    elif func1.func_type == FunctionType.X_OF_Y and func2.func_type == FunctionType.X_OF_Y:
        # For x=f(y) functions, swap roles of x and y and use the same approach
        y_values = find_intersections_y_of_x(func1, func2)
        return [(func1.evaluate(y), y) for y in y_values]
    
    elif (func1.func_type == FunctionType.Y_OF_X and func2.func_type == FunctionType.X_OF_Y) or \
         (func1.func_type == FunctionType.X_OF_Y and func2.func_type == FunctionType.Y_OF_X):
        # Mix of y=f(x) and x=g(y)
        if func1.func_type == FunctionType.X_OF_Y:
            func1, func2 = func2, func1  # Ensure func1 is y=f(x) and func2 is x=g(y)
        
        # For this case, we solve y = f(x) and x = g(y)
        # This creates an implicit system: y = f(x), y = g^(-1)(x)
        # We can use a numerical approach
        def system(vars):
            x, y = vars
            return [y - func1.evaluate(x), x - func2.evaluate(y)]
        
        # Try multiple starting points to find all intersections
        intersections = []
        for x_start in np.linspace(-5, 5, 10):
            for y_start in np.linspace(-5, 5, 10):
                try:
                    solution = optimize.root(system, [x_start, y_start])
                    if solution.success:
                        x, y = solution.x
                        if (-10 <= x <= 10 and -10 <= y <= 10 and 
                            np.isfinite(x) and np.isfinite(y) and
                            abs(y - func1.evaluate(x)) < 1e-5 and
                            abs(x - func2.evaluate(y)) < 1e-5):
                            intersections.append((round(x, 2), round(y, 2)))
                except:
                    continue
        
        # Remove duplicates
        unique_intersections = []
        for point in intersections:
            if all(abs(point[0]-p[0]) > 0.1 or abs(point[1]-p[1]) > 0.1 for p in unique_intersections):
                unique_intersections.append(point)
        
        return unique_intersections
    
    # Handle implicit and parametric functions
    # For simplicity, this implementation doesn't handle all possible combinations
    # A more complete implementation would use specialized techniques for each case
    return []

def create_function_plot(functions, points=None, x_min=-10, x_max=10, y_min=-10, y_max=10):
    """
    Create a plot of the functions
    
    Args:
        functions: List of Function objects
        points: Optional list of special points to mark (x, y, label)
        
    Returns:
        Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 6))
    
    # Track all valid y values to determine appropriate plot limits
    all_points = []
    
    # Plot each function
    for func in functions:
        try:
            if func.func_type == FunctionType.Y_OF_X:
                x, y = func.get_plot_data()
                plt.plot(x, y, label=f"f(x) = {func}")
                all_points.extend(zip(x, y))
                
            elif func.func_type == FunctionType.X_OF_Y:
                x, y = func.get_plot_data()
                plt.plot(x, y, label=f"x = {func}")
                all_points.extend(zip(x, y))
                
            elif func.func_type == FunctionType.IMPLICIT:
                X, Y, Z = func.get_plot_data()
                plt.contour(X, Y, Z, levels=[0], colors='blue')
                plt.text(0, 0, f"{func} = 0", fontsize=10, 
                         bbox=dict(facecolor='white', alpha=0.7))
                
            elif func.func_type == FunctionType.PARAMETRIC:
                x, y = func.get_plot_data()
                plt.plot(x, y, label=f"{func}")
                all_points.extend(zip(x, y))
        except Exception as e:
            print(f"Error plotting function {func}: {e}")
    
    # Plot special points and add them to points for limit calculation
    if points:
        for x, y, point_label in points:
            plt.plot(x, y, 'ro', markersize=6)
            plt.annotate(point_label, (x, y), xytext=(5, 5), textcoords='offset points')
            all_points.append((x, y))
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Set dynamically adjusted limits based on function values and points
    if all_points:
        x_values = [p[0] for p in all_points if np.isfinite(p[0])]
        y_values = [p[1] for p in all_points if np.isfinite(p[1])]
        
        if x_values:
            x_min_data = max(min(x_values), -100)
            x_max_data = min(max(x_values), 100)
            x_range = x_max_data - x_min_data
            plt.xlim(x_min_data - 0.1 * x_range, x_max_data + 0.1 * x_range)
        else:
            plt.xlim(x_min, x_max)
            
        if y_values:
            y_min_data = max(min(y_values), -100)
            y_max_data = min(max(y_values), 100)
            y_range = y_max_data - y_min_data
            plt.ylim(y_min_data - 0.1 * y_range, y_max_data + 0.1 * y_range)
        else:
            plt.ylim(y_min, y_max)
    else:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 string
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str

# --- Problem Generators ---

def generate_intersection_problem():
    """
    Generate a problem about finding the number of intersection points between two functions
    """
    # Decide on function types to use
    func_types = [
        (FunctionType.Y_OF_X, FunctionType.Y_OF_X),
        (FunctionType.X_OF_Y, FunctionType.X_OF_Y),
        (FunctionType.Y_OF_X, FunctionType.X_OF_Y),
        (FunctionType.IMPLICIT, FunctionType.Y_OF_X)
    ]
    chosen_types = random.choice(func_types)
    
    # Generate the functions based on chosen types
    if chosen_types[0] == FunctionType.Y_OF_X:
        generators = [
            lambda: generate_linear_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_quadratic_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_sin_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_cos_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_abs_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_rational_function(func_type=FunctionType.Y_OF_X)
        ]
        func1 = random.choice(generators)()
    elif chosen_types[0] == FunctionType.X_OF_Y:
        generators = [
            lambda: generate_linear_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_quadratic_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_sin_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_cos_function(func_type=FunctionType.X_OF_Y)
        ]
        func1 = random.choice(generators)()
    elif chosen_types[0] == FunctionType.IMPLICIT:
        generators = [
            generate_circle_function,
            generate_ellipse_function,
            generate_hyperbola_function,
            generate_parabola_implicit_function
        ]
        func1 = random.choice(generators)()
    
    if chosen_types[1] == FunctionType.Y_OF_X:
        generators = [
            lambda: generate_linear_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_quadratic_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_sin_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_cos_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_abs_function(func_type=FunctionType.Y_OF_X),
            lambda: generate_exp_function(func_type=FunctionType.Y_OF_X)
        ]
        func2 = random.choice(generators)()
    elif chosen_types[1] == FunctionType.X_OF_Y:
        generators = [
            lambda: generate_linear_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_quadratic_function(func_type=FunctionType.X_OF_Y),
            lambda: generate_abs_function(func_type=FunctionType.X_OF_Y)
        ]
        func2 = random.choice(generators)()
    
    # Find intersections
    intersection_points = find_intersections(func1, func2)
    
    # Create plot data
    plot_data = create_function_plot(
        [func1, func2], 
        points=[(x, y, f"({x}, {y})") for x, y in intersection_points]
    )
    
    # Create problem text
    templates = [
        f"How many points of intersection exist between the functions {func1} and {func2} in the range -10 ≤ x,y ≤ 10?",
        f"Find the number of times the graphs of the functions {func1} and {func2} intersect within the viewing window -10 ≤ x,y ≤ 10.",
        f"Determine the number of intersection points between the graphs defined by {func1} and {func2}."
    ]
    
    question = random.choice(templates)
    answer = len(intersection_points)
    
    return Problem(question=question, answer=answer, plot_data=plot_data)

def generate_circle_line_problem():
    """Generate a problem about finding the number of intersections between a circle and a line"""
    # Generate a circle
    radius = random.randint(2, 5)
    h = random.randint(-3, 3)
    k = random.randint(-3, 3)
    circle = generate_circle_function(radius=radius, center=(h, k))
    
    # Generate a line (as y=f(x))
    line = generate_linear_function(func_type=FunctionType.Y_OF_X)
    
    # A line can intersect a circle 0, 1, or 2 times
    # We need to convert the line to standard form ax + by + c = 0
    # For y = mx + b: -mx + y - b = 0
    
    # Extract line coefficients
    line_expr = line.expr_str
    if 'x' in line_expr:
        m = float(line_expr.split('x')[0].strip())
    else:
        m = 0
    
    if '+' in line_expr[-3:]:
        b = float(line_expr.split('+')[-1].strip())
    elif '-' in line_expr[-3:]:
        b = -float(line_expr.split('-')[-1].strip())
    else:
        b = 0
    
    # Distance from center to line: |mx - y + b|/sqrt(m^2 + 1)
    # Where (x,y) is the center of the circle (h,k)
    d = abs(m*h - k + b) / math.sqrt(m**2 + 1)
    
    # Determine number of intersections
    if d > radius:
        num_intersections = 0
    elif abs(d - radius) < 1e-9:
        num_intersections = 1
    else:
        num_intersections = 2
    
    # Find intersection points
    intersection_points = []
    
    def circle_line_system(vars):
        x, y = vars
        return [(x - h)**2 + (y - k)**2 - radius**2, y - (m*x + b)]
    
    # Try multiple starting points to find all intersections
    for x_start in np.linspace(-10, 10, 20):
        try:
            solution = optimize.root(circle_line_system, [x_start, m*x_start + b])
            if solution.success:
                x, y = solution.x
                if (-10 <= x <= 10 and -10 <= y <= 10 and 
                    abs((x - h)**2 + (y - k)**2 - radius**2) < 1e-5 and
                    abs(y - (m*x + b)) < 1e-5):
                    intersection_points.append((round(x, 2), round(y, 2)))
        except:
            continue
    
    # Remove duplicates
    unique_intersections = []
    for point in intersection_points:
        if all(abs(point[0]-p[0]) > 0.1 or abs(point[1]-p[1]) > 0.1 for p in unique_intersections):
            unique_intersections.append(point)
    
    # Create plot data
    plot_data = create_function_plot(
        [circle, line], 
        points=[(x, y, f"({x}, {y})") for x, y in unique_intersections]
    )
    
    # Create problem text
    templates = [
        f"How many points of intersection exist between the circle {circle} and the line {line}?",
        f"Find the number of times the line {line} intersects the circle {circle}.",
        f"Determine the number of intersection points between the circle {circle} and the line {line}."
    ]
    
    question = random.choice(templates)
    answer = num_intersections
    
    return Problem(question=question, answer=answer, plot_data=plot_data)

def generate_conics_intersection_problem():
    """Generate a problem about the number of intersections between two conic sections"""
    # Choose two different conic generators
    conic_generators = [
        generate_circle_function,
        generate_ellipse_function,
        generate_hyperbola_function,
        generate_parabola_implicit_function
    ]
    
    gen1, gen2 = random.sample(conic_generators, 2)
    
    # Generate the conics
    conic1 = gen1()
    conic2 = gen2()
    
    # Finding intersections between implicit functions is challenging
    # We'll use a numerical approach to estimate the number of intersections
    # by sampling points in the grid and checking for sign changes in both functions
    
    grid_size = 100
    x_vals = np.linspace(-10, 10, grid_size)
    y_vals = np.linspace(-10, 10, grid_size)
    
    # Create grids for sampling
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Evaluate both functions on the grid
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z1[i, j] = conic1.evaluate(X[i, j], Y[i, j])
                Z2[i, j] = conic2.evaluate(X[i, j], Y[i, j])
            except:
                Z1[i, j] = np.nan
                Z2[i, j] = np.nan
    
    # Find points where both functions are close to zero
    # This is an estimate of intersection points
    intersection_points = []
    threshold = 0.1
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if abs(Z1[i, j]) < threshold and abs(Z2[i, j]) < threshold:
                x, y = X[i, j], Y[i, j]
                # Avoid duplicates by checking if point is close to existing ones
                is_duplicate = False
                for px, py in intersection_points:
                    if abs(px - x) < 0.2 and abs(py - y) < 0.2:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Refine the solution using optimization
                    def system(vars):
                        x, y = vars
                        return [conic1.evaluate(x, y), conic2.evaluate(x, y)]
                    
                    try:
                        solution = optimize.root(system, [x, y])
                        if solution.success:
                            x_sol, y_sol = solution.x
                            if (-10 <= x_sol <= 10 and -10 <= y_sol <= 10 and
                                abs(conic1.evaluate(x_sol, y_sol)) < threshold and
                                abs(conic2.evaluate(x_sol, y_sol)) < threshold):
                                intersection_points.append((round(x_sol, 2), round(y_sol, 2)))
                    except:
                        # If optimization fails, use the original grid point
                        intersection_points.append((round(x, 2), round(y, 2)))
    
    # Create plot data
    plot_data = create_function_plot(
        [conic1, conic2], 
        points=[(x, y, f"({x}, {y})") for x, y in intersection_points]
    )
    
    # Create problem text
    templates = [
        f"How many points of intersection exist between {conic1} and {conic2} in the range -10 ≤ x,y ≤ 10?",
        f"Find the number of times the curves {conic1} and {conic2} intersect within the viewing window.",
        f"Determine the number of intersection points between the conics defined by {conic1} and {conic2}."
    ]
    
    question = random.choice(templates)
    answer = len(intersection_points)
    
    return Problem(question=question, answer=answer, plot_data=plot_data)

def generate_parametric_problem():
    """Generate a problem about a parametric curve"""
    # Choose a parametric function generator
    parametric_generators = [
        generate_parametric_circle,
        generate_parametric_ellipse,
        generate_parametric_lissajous,
        generate_spiral_function
    ]
    
    generator = random.choice(parametric_generators)
    func = generator()
    
    # Questions we can ask depends on the type of parametric function
    if func.expr_x_str.startswith(generate_parametric_circle.__name__) or \
       func.expr_x_str.startswith(generate_parametric_ellipse.__name__):
        # For circles and ellipses, we can ask about circumference/perimeter
        
        # Extract parameters from expressions
        if "cos" in func.expr_x_str and "sin" in func.expr_y_str:
            # Circle case
            if func.expr_x_str.count("cos") == 1:
                radius = float(func.expr_x_str.split("cos")[0].split("+")[-1].strip())
                circumference = 2 * np.pi * radius
                
                templates = [
                    f"Find the circumference of the curve given by the parametric equations {func}.",
                    f"Calculate the circumference of the circle described by the parametric equations {func}.",
                    f"Determine the distance traveled around the parametric curve {func} for one complete cycle of the parameter t."
                ]
                
                question = random.choice(templates)
                answer = round(circumference, 2)
            
            # Ellipse case
            else:
                a = float(func.expr_x_str.split("cos")[0].split("+")[-1].strip())
                b = float(func.expr_y_str.split("sin")[0].split("+")[-1].strip())
                
                # Approximation of ellipse perimeter
                perimeter = 2 * np.pi * np.sqrt((a**2 + b**2) / 2)
                
                templates = [
                    f"Find the perimeter of the ellipse given by the parametric equations {func}. Round to 2 decimal places.",
                    f"Calculate the perimeter of the elliptical curve described by the parametric equations {func}. Round to 2 decimal places.",
                    f"Determine the distance traveled around the parametric curve {func} for one complete cycle of t. Round to 2 decimal places."
                ]
                
                question = random.choice(templates)
                answer = round(perimeter, 2)
        
        else:
            # Fallback - ask about the domain
            t_min, t_max = func.domain
            domain_length = t_max - t_min
            
            question = f"What is the length of the parameter domain for the parametric curve {func}?"
            answer = round(domain_length, 2)
    
    else:
        # For other curves, ask about the domain or number of self-intersections
        t_min, t_max = func.domain
        
        # Check if it's a closed curve (returns to starting point)
        x_start, y_start = func.evaluate(t_min)
        x_end, y_end = func.evaluate(t_max)
        
        is_closed = (abs(x_start - x_end) < 0.1 and abs(y_start - y_end) < 0.1)
        
        if is_closed:
            question = f"Is the parametric curve {func} closed (returns to its starting point)? Answer 'yes' or 'no'."
            answer = "yes"
        else:
            question = f"What is the length of the parameter domain for the parametric curve {func}?"
            answer = round(t_max - t_min, 2)
    
    # Create plot data
    plot_data = create_function_plot([func], points=[])
    
    return Problem(question=question, answer=answer, plot_data=plot_data)

def generate_function_derivative_problem():
    """Generate a problem about finding the derivative of a function at a point"""
    # Generate a function (only y=f(x) type supports derivatives easily)
    generators = [
        lambda: generate_linear_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_quadratic_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_sin_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_cos_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_abs_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_exp_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_composite_function_y_of_x()
    ]
    
    func = random.choice(generators)()
    
    # Define a numerical derivative function
    def derivative(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    # Choose a point at which to evaluate the derivative
    x_point = random.randint(-5, 5)
    
    try:
        # Calculate the derivative at the chosen point
        derivative_value = derivative(func.func, x_point)
        
        # Ensure the derivative value is reasonable
        if not np.isfinite(derivative_value) or abs(derivative_value) > 100:
            # Try a different point if derivative is too large or undefined
            x_point = random.choice([-3, -2, -1, 0, 1, 2, 3])
            derivative_value = derivative(func.func, x_point)
            
            # If still problematic, use a simpler function
            if not np.isfinite(derivative_value) or abs(derivative_value) > 100:
                func = generate_linear_function(func_type=FunctionType.Y_OF_X)
                derivative_value = derivative(func.func, x_point)
    except:
        # If an error occurs, fall back to a linear function
        func = generate_linear_function(func_type=FunctionType.Y_OF_X)
        derivative_value = derivative(func.func, x_point)
    
    # Create plot data
    y_point = func.func(x_point)
    
    # Create a plot of the function with the tangent line at the chosen point
    def tangent_line(x):
        return y_point + derivative_value * (x - x_point)
    
    tangent_func = FunctionYofX(tangent_line, f"{y_point} + {derivative_value:.2f}·(x - {x_point})")
    
    plot_data = create_function_plot(
        [func, tangent_func], 
        points=[(x_point, y_point, f"({x_point}, {y_point:.2f})")]
    )
    
    # Round the derivative to 2 decimal places
    derivative_value = round(derivative_value, 2)
    
    templates = [
        f"Find the derivative of f(x) = {func} at x = {x_point}. Round to 2 decimal places.",
        f"Calculate the slope of the tangent line to the curve y = {func} at the point where x = {x_point}. Round to 2 decimal places.",
        f"What is the instantaneous rate of change of the function f(x) = {func} at x = {x_point}? Round to 2 decimal places."
    ]
    
    question = random.choice(templates)
    answer = derivative_value
    
    return Problem(question=question, answer=answer, plot_data=plot_data)

def generate_function_properties_problem():
    """Generate a problem about various properties of a function"""
    # Generate a function with interesting properties
    func_generators = [
        lambda: generate_quadratic_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_sin_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_cos_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_abs_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_rational_function(func_type=FunctionType.Y_OF_X),
        lambda: generate_composite_function_y_of_x()
    ]
    
    func = random.choice(func_generators)()
    
    # Define a numerical derivative function
    def derivative(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)
    
    # Sample points to analyze function properties
    x_samples = np.linspace(-10, 10, 1000)
    
    # Collect valid function values
    valid_x = []
    valid_y = []
    
    for x in x_samples:
        try:
            y = func.func(x)
            if np.isfinite(y) and -1000 < y < 1000:
                valid_x.append(x)
                valid_y.append(y)
        except:
            continue
    
    if not valid_x:
        # If no valid points, use a simpler function
        func = generate_quadratic_function(func_type=FunctionType.Y_OF_X)
        valid_x = []
        valid_y = []
        
        for x in x_samples:
            try:
                y = func.func(x)
                if np.isfinite(y) and -1000 < y < 1000:
                    valid_x.append(x)
                    valid_y.append(y)
            except:
                continue
    
    # Find extrema (local minima and maxima)
    extrema = []
    for i in range(1, len(valid_x)-1):
        x_prev, y_prev = valid_x[i-1], valid_y[i-1]
        x, y = valid_x[i], valid_y[i]
        x_next, y_next = valid_x[i+1], valid_y[i+1]
        
        if (y_prev < y > y_next) or (y_prev > y < y_next):
            # Verify with derivative test
            try:
                deriv = derivative(func.func, x)
                if abs(deriv) < 0.1:  # Close to zero
                    extrema_type = "Maximum" if y_prev < y > y_next else "Minimum"
                    extrema.append((x, y, extrema_type))
            except:
                continue
    
    # Find zeros by looking for sign changes
    zeros = []
    for i in range(1, len(valid_y)):
        if valid_y[i-1] * valid_y[i] <= 0 and abs(valid_y[i-1]) + abs(valid_y[i]) > 1e-10:
            # Linear interpolation to find the zero
            x0, y0 = valid_x[i-1], valid_y[i-1]
            x1, y1 = valid_x[i], valid_y[i]
            
            if abs(y1 - y0) > 1e-10:
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                zeros.append(round(x_zero, 2))
    
    # Create plot data
    plot_points = []
    for x, y, label in extrema:
        plot_points.append((x, y, f"{label} ({round(x, 2)}, {round(y, 2)})"))
    
    for x in zeros:
        plot_points.append((x, 0, f"Zero ({x}, 0)"))
    
    plot_data = create_function_plot([func], points=plot_points)
    
    # Choose which property to ask about
    properties = []
    
    if extrema:
        properties.append("num_extrema")
        properties.append("min_extrema_x")
        properties.append("max_extrema_x")
    
    if zeros:
        properties.append("num_zeros")
        properties.append("leftmost_zero")
        properties.append("rightmost_zero")
    
    # If function has interesting properties like periodicity or symmetry
    if isinstance(func, FunctionYofX) and ("sin" in func.expr_str or "cos" in func.expr_str):
        properties.append("periodicity")
        properties.append("symmetry")
    
    # Choose randomly from available properties
    if properties:
        property_type = random.choice(properties)
        
        if property_type == "num_extrema":
            question = f"How many local extrema (minima and maxima combined) does the function f(x) = {func} have in the interval [-10, 10]?"
            answer = len(extrema)
        
        elif property_type == "min_extrema_x":
            # Find x-coordinate of minimum with lowest y-value
            minima = [(x, y) for x, y, typ in extrema if typ == "Minimum"]
            if minima:
                x, y = min(minima, key=lambda p: p[1])
                question = f"Find the x-coordinate of the local minimum with the smallest y-value for the function f(x) = {func} in the interval [-10, 10]. Round to 2 decimal places."
                answer = round(x, 2)
            else:
                question = f"How many local minima does the function f(x) = {func} have in the interval [-10, 10]?"
                answer = 0
        
        elif property_type == "max_extrema_x":
            # Find x-coordinate of maximum with highest y-value
            maxima = [(x, y) for x, y, typ in extrema if typ == "Maximum"]
            if maxima:
                x, y = max(maxima, key=lambda p: p[1])
                question = f"Find the x-coordinate of the local maximum with the largest y-value for the function f(x) = {func} in the interval [-10, 10]. Round to 2 decimal places."
                answer = round(x, 2)
            else:
                question = f"How many local maxima does the function f(x) = {func} have in the interval [-10, 10]?"
                answer = 0
        
        elif property_type == "num_zeros":
            question = f"How many x-intercepts (zeros) does the function f(x) = {func} have in the interval [-10, 10]?"
            answer = len(zeros)
        
        elif property_type == "leftmost_zero":
            if zeros:
                question = f"Find the leftmost x-intercept (zero) of the function f(x) = {func} in the interval [-10, 10]."
                answer = min(zeros)
            else:
                question = f"How many x-intercepts (zeros) does the function f(x) = {func} have in the interval [-10, 10]?"
                answer = 0
        
        elif property_type == "rightmost_zero":
            if zeros:
                question = f"Find the rightmost x-intercept (zero) of the function f(x) = {func} in the interval [-10, 10]."
                answer = max(zeros)
            else:
                question = f"How many x-intercepts (zeros) does the function f(x) = {func} have in the interval [-10, 10]?"
                answer = 0
        
        elif property_type == "periodicity":
            # For simplicity, estimate the period from the function expression
            if "sin" in func.expr_str or "cos" in func.expr_str:
                # Extract frequency from expression
                try:
                    if "sin(" in func.expr_str:
                        freq_str = func.expr_str.split("sin(")[1].split("πx")[0]
                    else:  # cos case
                        freq_str = func.expr_str.split("cos(")[1].split("πx")[0]
                    
                    if freq_str:
                        freq = float(freq_str)
                        period = 2 / freq
                    else:
                        period = 2  # Default for sin(πx) or cos(πx)
                except:
                    period = 2  # Default fallback
                
                question = f"Find the period of the function f(x) = {func}. Round to 2 decimal places if necessary."
                answer = round(period, 2)
            else:
                # Fall back to extrema question if periodicity can't be determined
                question = f"How many local extrema (minima and maxima combined) does the function f(x) = {func} have in the interval [-10, 10]?"
                answer = len(extrema)
        
        elif property_type == "symmetry":
            # Check for even/odd symmetry numerically
            even_test = sum(abs(func.func(x) - func.func(-x)) for x in np.linspace(0.5, 5, 10) if np.isfinite(func.func(x)) and np.isfinite(func.func(-x)))
            odd_test = sum(abs(func.func(x) + func.func(-x)) for x in np.linspace(0.5, 5, 10) if np.isfinite(func.func(x)) and np.isfinite(func.func(-x)))
            
            # Determine symmetry based on numerical tests
            if even_test < 0.1:
                symmetry = "even"
            elif odd_test < 0.1:
                symmetry = "odd"
            else:
                symmetry = "neither"
            
            question = f"Is the function f(x) = {func} even, odd, or neither? (Answer 'even', 'odd', or 'neither')"
            answer = symmetry
    else:
        # If no interesting properties, ask about function value at a point
        x_point = random.randint(-5, 5)
        y_value = func.func(x_point)
        
        question = f"Calculate f({x_point}) for the function f(x) = {func}. Round to 2 decimal places if necessary."
        answer = round(y_value, 2)
    
    return Problem(question=question, answer=answer, plot_data=plot_data)

# --- Main Function and CLI Interface ---

def main():
    """
    Command-line interface for generating enhanced function analysis problems.
    """
    parser = argparse.ArgumentParser(description='Generate enhanced function analysis problems.')
    parser.add_argument('--type', type=str,
                      choices=[
                          'intersection', 
                          'circle_line', 
                          'conics_intersection',
                          'parametric', 
                          'derivative',
                          'function_properties',
                          'all'
                      ], 
                      default='all', help='Type of function problem to generate')
    parser.add_argument('--count', type=int, default=10, help='Number of problems to generate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--save_plots', default=True, help='Save plots to files')
    parser.add_argument('--plot_dir', type=str, default='enhanced_plots', help='Directory to save plots')
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.save_plots:
        import os
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)

    if args.type == 'all':
        problem_types = [
            'intersection', 
            'circle_line', 
            'conics_intersection',
            'parametric', 
            'derivative',
            'function_properties'
        ]
    else:
        problem_types = [args.type]

    for _ in range(args.count):
        for problem_type in problem_types:
            print(f"\n--- {problem_type.upper()} PROBLEM ---")
            
            if problem_type == 'intersection':
                problem = generate_intersection_problem()
            elif problem_type == 'circle_line':
                problem = generate_circle_line_problem()
            elif problem_type == 'conics_intersection':
                problem = generate_conics_intersection_problem()
            elif problem_type == 'parametric':
                problem = generate_parametric_problem()
            elif problem_type == 'derivative':
                problem = generate_function_derivative_problem()
            elif problem_type == 'function_properties':
                problem = generate_function_properties_problem()
            
            print(f"Question: {problem.question}")
            print(f"\nAnswer: {problem.answer}")
            
            if args.save_plots and problem.plot_data:
                # Save the plot
                plot_file = f"{args.plot_dir}/{problem_type}_{_}.png"
                with open(plot_file, 'wb') as f:
                    f.write(base64.b64decode(problem.plot_data))
                print(f"Plot saved to {plot_file}")
            
            print("-" * 50)


if __name__ == "__main__":
    main() 