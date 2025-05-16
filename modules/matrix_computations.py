"""
Matrix computations module containing various matrix problem generators.
This module generates problems related to matrix inverse, matrix multiplication,
matrix exponential, SVD decomposition, matrix determinant, and matrix rank.
"""

import numpy as np
import random
import argparse
import math
from collections import namedtuple
from scipy import linalg

# Simple Problem container
Problem = namedtuple('Problem', ('question', 'answer'))

# --- Matrix Generation Functions ---

def generate_matrix(rows, cols, min_val=-10, max_val=10, integer_only=True, 
                    special_type=None, singular=False, positive_definite=False):
    """
    Generate a matrix with specified dimensions and properties.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        integer_only: If True, generate integer values only
        special_type: None, 'symmetric', 'diagonal', 'triangular', 'orthogonal'
        singular: If True, ensure the matrix is singular (for square matrices)
        positive_definite: If True, generate a positive definite matrix (symmetric)
        
    Returns:
        numpy.ndarray: Generated matrix
    """
    # Special types require square matrices
    if special_type in ['symmetric', 'diagonal', 'orthogonal'] or positive_definite:
        if rows != cols:
            rows = cols = max(rows, cols)
    
    if special_type == 'orthogonal':
        # Generate an orthogonal matrix using QR decomposition
        A = np.random.randn(rows, cols)
        Q, R = np.linalg.qr(A)
        # Round for integer orthogonal matrices
        if integer_only:
            Q = np.round(Q).astype(int)
        return Q
    
    if special_type == 'diagonal':
        # Generate a diagonal matrix
        if integer_only:
            diag_elements = np.random.randint(min_val, max_val+1, size=rows)
            if singular and rows > 1:
                # Make the matrix singular by setting one diagonal element to zero
                diag_elements[random.randint(0, rows-1)] = 0
        else:
            diag_elements = np.random.uniform(min_val, max_val, size=rows)
            if singular and rows > 1:
                diag_elements[random.randint(0, rows-1)] = 0
        
        return np.diag(diag_elements)
    
    if positive_definite:
        # Generate a positive definite matrix
        A = np.random.randn(rows, cols)
        matrix = A.dot(A.T) + np.eye(rows) * rows  # Ensure it's positive definite
        if integer_only:
            return np.round(matrix).astype(int)
        return matrix
    
    # Generate a base matrix
    if integer_only:
        matrix = np.random.randint(min_val, max_val+1, size=(rows, cols))
    else:
        matrix = np.random.uniform(min_val, max_val, size=(rows, cols))
    
    if special_type == 'symmetric':
        # Convert to symmetric
        matrix = (matrix + matrix.T) / 2
        if integer_only:
            matrix = np.round(matrix).astype(int)
    
    elif special_type == 'triangular':
        # Convert to upper triangular
        matrix = np.triu(matrix)
    
    if singular and rows == cols and rows > 1:
        # Make the matrix singular by setting a row to a multiple of another row
        row_to_replace = random.randint(0, rows-1)
        row_to_copy = (row_to_replace + 1) % rows
        multiplier = random.randint(1, 3)
        matrix[row_to_replace] = multiplier * matrix[row_to_copy]
    
    return matrix

def format_matrix(matrix, decimals=2):
    """Format a matrix for display in problem statements"""
    rows, cols = matrix.shape
    formatted = "["
    for i in range(rows):
        formatted += "["
        for j in range(cols):
            value = matrix[i, j]
            # Handle the formatting of integer vs. float values
            if np.isclose(value, round(value)):
                formatted += f"{int(round(value))}"
            else:
                formatted += f"{value:.{decimals}f}"
            if j < cols - 1:
                formatted += ", "
        formatted += "]"
        if i < rows - 1:
            formatted += ",\n "
    formatted += "]"
    return formatted

# --- Matrix Problem Generator Functions ---

def generate_matrix_inverse_problem(rows=None, special_type=None, min_val=-5, max_val=5):
    """
    Generate a matrix inverse problem with an integer answer.
    
    Args:
        rows: Size of the square matrix
        special_type: Type of matrix to generate
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    if rows is None:
        rows = random.choice([2, 3])
    
    # Generate an invertible matrix
    while True:
        matrix = generate_matrix(rows, rows, min_val, max_val, integer_only=True, special_type=special_type)
        if np.linalg.det(matrix) != 0:
            break
    
    # Scale down the matrix to make the inverse have larger values
    # Dynamically set scaling factor based on matrix size
    # Smaller matrices need less scaling, larger ones need more
    scaling_factor = 1.0 / (rows * 2 * max(abs(max_val), abs(min_val)))  # Scaling gets smaller as rows increase
    scaled_matrix = matrix * scaling_factor
    
    # Calculate the inverse
    inverse = np.linalg.inv(scaled_matrix)
    
    # Calculate an integer value from the inverse matrix
    calculation_types = [
        "sum of all entries",
        "sum of absolute values of all entries",
        "trace",
        "largest entry by absolute value",
        "entry at position"
    ]
    
    calc_type = random.choice(calculation_types)
    
    if calc_type == "sum of all entries":
        result = int(round(np.sum(inverse)))
        calc_description = "the sum of all entries"
    elif calc_type == "sum of absolute values of all entries":
        result = int(round(np.sum(np.abs(inverse))))
        calc_description = "the sum of absolute values of all entries"
    elif calc_type == "trace":
        result = int(round(np.trace(inverse)))
        calc_description = "the trace (sum of diagonal elements)"
    elif calc_type == "largest entry by absolute value":
        max_idx = np.unravel_index(np.argmax(np.abs(inverse)), inverse.shape)
        result = int(round(inverse[max_idx]))
        calc_description = f"the value of the entry with the largest absolute value"
    else:  # entry at position
        i, j = random.randint(0, rows-1), random.randint(0, rows-1)
        result = int(round(inverse[i, j]))
        calc_description = f"the value at position ({i+1}, {j+1})"
    
    # Use the original matrix in the problem statement
    matrix_str = format_matrix(matrix)
    
    scaling_display = f"1/{int(1/scaling_factor)}"

    templates = [
        f"Find the inverse of {scaling_display} times the matrix and calculate {calc_description} of the inverse. Round your answer to the nearest integer:\n{matrix_str}",
        f"Calculate the inverse matrix of {scaling_display} times the following and determine {calc_description} of the resulting matrix. Your answer should be rounded to the nearest integer:\n{matrix_str}",
        f"Find {calc_description} of the inverse of matrix {scaling_display} * A where A =\n{matrix_str}\nRound your answer to the nearest integer.",
        f"For the inverse of {scaling_display} times the matrix below, what is {calc_description} rounded to the nearest integer?\n{matrix_str}"
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=result)

def generate_matrix_multiplication_problem(rows_a=None, cols_a=None, cols_b=None, min_val=-5, max_val=5):
    """
    Generate a matrix multiplication problem with an integer answer.
    
    Args:
        rows_a: Number of rows for the first matrix
        cols_a: Number of columns for the first matrix
        cols_b: Number of columns for the second matrix
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    # Set default dimensions if not provided
    if rows_a is None:
        rows_a = random.randint(2, 4)
    if cols_a is None:
        cols_a = random.randint(2, 4)
    if cols_b is None:
        cols_b = random.randint(2, 4)
    
    # For matrix multiplication, rows_b must equal cols_a
    rows_b = cols_a
    
    # Generate matrices
    matrix_a = generate_matrix(rows_a, cols_a, min_val, max_val, integer_only=True)
    matrix_b = generate_matrix(rows_b, cols_b, min_val, max_val, integer_only=True)
    
    # Calculate the product
    product = np.matmul(matrix_a, matrix_b)
    
    # Calculate an integer value from the product matrix
    calculation_types = [
        "sum of all entries",
        "trace if it's square",
        "specific entry",
        "sum of absolute values"
    ]
    
    calc_type = random.choice(calculation_types)
    
    if calc_type == "trace if it's square" and product.shape[0] == product.shape[1]:
        result = int(np.trace(product))
        calc_description = "the trace (sum of diagonal elements)"
    elif calc_type == "specific entry":
        i, j = random.randint(0, product.shape[0]-1), random.randint(0, product.shape[1]-1)
        result = int(product[i, j])
        calc_description = f"the value at position ({i+1}, {j+1})"
    elif calc_type == "sum of absolute values":
        result = int(np.sum(np.abs(product)))
        calc_description = "the sum of absolute values of all entries"
    else:  # sum of all entries
        result = int(np.sum(product))
        calc_description = "the sum of all entries"
    
    # Format matrices for display
    matrix_a_str = format_matrix(matrix_a)
    matrix_b_str = format_matrix(matrix_b)
    
    templates = [
        f"Calculate the matrix product AB and find {calc_description} of the result:\nA =\n{matrix_a_str}\nB =\n{matrix_b_str}",
        f"Find {calc_description} of the product of the following matrices:\nA =\n{matrix_a_str}\nB =\n{matrix_b_str}",
        f"What is {calc_description} when matrix A is multiplied by matrix B?\nA =\n{matrix_a_str}\nB =\n{matrix_b_str}",
        f"Compute {calc_description} of the product matrix AB given:\nA =\n{matrix_a_str}\nB =\n{matrix_b_str}"
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=result)

def generate_matrix_exponential_problem(rows=None, special_type=None, min_val=-3, max_val=3):
    """
    Generate a matrix exponential problem with an integer answer.
    
    Args:
        rows: Size of the square matrix
        special_type: Type of matrix to generate
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    if rows is None:
        rows = random.choice([2, 3])
    
    # Generate a matrix with small values to avoid overflow
    matrix = generate_matrix(rows, rows, min_val, max_val, integer_only=True, 
                           special_type=special_type)
    
    # Calculate the matrix exponential
    exp_matrix = linalg.expm(matrix)
    
    # Calculate an integer value from the exponential matrix
    calculation_types = [
        "trace",
        "sum of all entries",
        "sum of absolute values",
        "specific entry"
    ]
    
    calc_type = random.choice(calculation_types)
    
    if calc_type == "trace":
        result = int(round(np.trace(exp_matrix)))
        calc_description = "the trace (sum of diagonal elements)"
    elif calc_type == "sum of all entries":
        result = int(round(np.sum(exp_matrix)))
        calc_description = "the sum of all entries"
    elif calc_type == "sum of absolute values":
        result = int(round(np.sum(np.abs(exp_matrix))))
        calc_description = "the sum of absolute values of all entries"
    else:  # specific entry
        i, j = random.randint(0, rows-1), random.randint(0, rows-1)
        result = int(round(exp_matrix[i, j]))
        calc_description = f"the value at position ({i+1}, {j+1})"
    
    matrix_str = format_matrix(matrix)
    
    templates = [
        f"Calculate the matrix exponential e^A and find {calc_description} of the result:\nA =\n{matrix_str}",
        f"Find {calc_description} of the exponential of the matrix:\n{matrix_str}",
        f"What is {calc_description} of the matrix e^A where A is:\n{matrix_str}?",
        f"For the matrix exponential e^A where A =\n{matrix_str}, determine {calc_description}."
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=result)

def generate_svd_decomposition_problem(rows=None, cols=None, min_val=-5, max_val=5):
    """
    Generate an SVD decomposition problem with an integer answer.
    
    Args:
        rows: Number of rows for the matrix
        cols: Number of columns for the matrix
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    if rows is None:
        rows = random.choice([2, 3, 4])
    
    if cols is None:
        cols = random.choice([2, 3, 4])
    
    # Generate a matrix
    matrix = generate_matrix(rows, cols, min_val, max_val, integer_only=True)
    
    # Calculate the SVD
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # Calculate an integer value from the SVD
    calculation_types = [
        "largest singular value",
        "sum of singular values",
        "difference between largest and smallest singular values"
    ]
    
    calc_type = random.choice(calculation_types)
    
    if calc_type == "largest singular value":
        result = int(round(S[0]))
        calc_description = "the rounded largest singular value"
    elif calc_type == "sum of singular values":
        result = int(round(np.sum(S)))
        calc_description = "the rounded sum of all singular values"
    else:  # difference
        result = int(round(S[0] - S[-1]))
        calc_description = "the rounded difference between the largest and smallest singular values"
    
    matrix_str = format_matrix(matrix)
    
    templates = [
        f"Find the Singular Value Decomposition (SVD) of the matrix and determine {calc_description}:\n{matrix_str}",
        f"Calculate {calc_description} from the SVD of:\n{matrix_str}",
        f"What is {calc_description} in the SVD of the following matrix?\n{matrix_str}",
        f"Compute {calc_description} from the singular value decomposition of the matrix:\n{matrix_str}"
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=result)

def generate_matrix_determinant_problem(rows=None, special_type=None, min_val=-5, max_val=5):
    """
    Generate a matrix determinant problem with an integer answer modulo 1000.
    
    Args:
        rows: Size of the square matrix
        special_type: Type of matrix to generate
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        
    Returns:
        Problem: A Problem object with question and answer as a single integer (det % 1000)
    """
    if rows is None:
        rows = random.choice([2, 3, 4])
    
    # Generate a matrix with integer determinant
    while True:
        matrix = generate_matrix(rows, rows, min_val, max_val, integer_only=True, 
                               special_type=special_type)
        det = int(round(np.linalg.det(matrix)))
        # Ensure determinant is a reasonable integer
        if abs(np.linalg.det(matrix) - det) < 1e-10 and det != 0:
            break
    
    matrix_str = format_matrix(matrix)
    det_mod = det % 1000
    
    templates = [
        f"Calculate the determinant of the matrix, round to the nearest integer, then find the result modulo 1000:\n{matrix_str}",
        f"Find the determinant of the following matrix. Round to the nearest integer, then give your answer modulo 1000:\n{matrix_str}",
        f"Compute |A| for the matrix A =\n{matrix_str}\nRound to the nearest integer, then take the result modulo 1000.",
        f"What is the determinant of the matrix:\n{matrix_str}? Round to the nearest integer, then express your answer modulo 1000.",
        f"Determine the value of det(A) where A =\n{matrix_str}\nRound to the nearest integer, then give your answer as det(A) mod 1000."
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=det_mod)

def generate_matrix_rank_problem(rows=None, cols=None, min_val=-5, max_val=5, specific_rank=None):
    """
    Generate a matrix rank problem with an integer answer.
    
    Args:
        rows: Number of rows for the matrix
        cols: Number of columns for the matrix
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        specific_rank: If provided, generate a matrix with this specific rank
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    if rows is None:
        rows = random.choice([3, 4, 5])
    
    if cols is None:
        cols = random.choice([3, 4, 5])
    
    max_possible_rank = min(rows, cols)
    
    if specific_rank is not None and specific_rank <= max_possible_rank:
        # Generate a matrix with a specific rank using matrix factorization
        # For a matrix with rank r, we can create it as a product of an (rows × r) and (r × cols) matrix
        
        # Create the left matrix (rows × specific_rank)
        left_matrix = np.random.randint(min_val, max_val+1, size=(rows, specific_rank))
        
        # Create the right matrix (specific_rank × cols)
        right_matrix = np.random.randint(min_val, max_val+1, size=(specific_rank, cols))
        
        # Multiply to get our matrix with the desired rank
        matrix = np.matmul(left_matrix, right_matrix)
        
        rank = specific_rank
    else:
        # Generate a random matrix
        matrix = generate_matrix(rows, cols, min_val, max_val, integer_only=True)
        
        # Calculate its rank
        rank = np.linalg.matrix_rank(matrix)
    
    matrix_str = format_matrix(matrix)
    
    templates = [
        f"Find the rank of the matrix:\n{matrix_str}",
        f"Determine the rank of the following matrix:\n{matrix_str}",
        f"Calculate the rank of matrix A where A =\n{matrix_str}",
        f"What is the rank of the matrix:\n{matrix_str}?",
        f"Compute the rank of the given {rows}x{cols} matrix:\n{matrix_str}"
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=rank)

def generate_eigenvalue_problem(rows=None, special_type=None, min_val=-5, max_val=5):
    """
    Generate a matrix eigenvalue problem with an integer answer.
    
    Args:
        rows: Size of the square matrix
        special_type: Type of matrix to generate
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    if rows is None:
        rows = random.choice([2, 3, 4])
    
    # Use symmetric matrices for guaranteed real eigenvalues
    if special_type is None:
        special_type = 'symmetric'
    
    # Generate a matrix
    matrix = generate_matrix(rows, rows, min_val, max_val, integer_only=True, 
                           special_type=special_type)
    
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Calculate an integer value from the eigenvalues
    calculation_types = [
        "largest eigenvalue",
        # "sum of eigenvalues",
        # "product of eigenvalues",
        # "number of positive eigenvalues",
        # "number of negative eigenvalues",
        "sum of absolute values of eigenvalues"
    ]
    
    calc_type = random.choice(calculation_types)
    
    if calc_type == "largest eigenvalue":
        result = int(round(np.max(np.abs(eigenvalues))))
        calc_description = "the largest eigenvalue by absolute value"
    elif calc_type == "sum of eigenvalues":
        result = int(round(np.sum(eigenvalues)))
        calc_description = "the sum of all eigenvalues"
    elif calc_type == "product of eigenvalues":
        result = int(round(np.prod(eigenvalues)))
        calc_description = "the product of all eigenvalues"
    elif calc_type == "number of positive eigenvalues":
        result = int(np.sum(eigenvalues > 1e-10))
        calc_description = "the number of positive eigenvalues"
    elif calc_type == "number of negative eigenvalues":
        result = int(np.sum(eigenvalues < -1e-10))
        calc_description = "the number of negative eigenvalues"
    else:  # sum of absolute values
        result = int(round(np.sum(np.abs(eigenvalues))))
        calc_description = "the sum of absolute values of all eigenvalues"
    
    matrix_str = format_matrix(matrix)
    
    templates = [
        f"Find the eigenvalues of the matrix and calculate {calc_description}:\n{matrix_str}",
        f"Calculate {calc_description} for the matrix:\n{matrix_str}",
        f"What is {calc_description} for the following matrix?\n{matrix_str}",
        f"Determine {calc_description} of the matrix:\n{matrix_str}"
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=result)

def generate_matrix_power_problem(rows=None, special_type=None, min_val=-5, max_val=5,
                              min_power=2, max_power=4):
    """
    Generate a matrix power problem (A^n) with an integer answer.
    
    Args:
        rows: Size of the square matrix
        special_type: Type of matrix to generate
        min_val: Minimum value for matrix elements
        max_val: Maximum value for matrix elements
        min_power: Minimum power to raise the matrix to
        max_power: Maximum power to raise the matrix to
        
    Returns:
        Problem: A Problem object with question and answer as a single integer
    """
    if rows is None:
        rows = random.choice([2, 3])
    
    # Generate a square matrix with small values to avoid overflow
    matrix = generate_matrix(rows, rows, min_val, max_val, integer_only=True, 
                           special_type=special_type)
    
    # Choose a power between min_power and max_power
    power = random.randint(min_power, max_power)
    
    # Calculate the matrix power
    power_matrix = np.linalg.matrix_power(matrix, power)
    
    # Calculate an integer value from the powered matrix
    calculation_types = [
        # "trace",
        "sum of all entries",
        "sum of absolute values",
        "specific entry"
    ]
    
    calc_type = random.choice(calculation_types)
    
    if calc_type == "trace":
        result = int(round(np.trace(power_matrix)))
        calc_description = "the trace (sum of diagonal elements)"
    elif calc_type == "sum of all entries":
        result = int(round(np.sum(power_matrix)))
        calc_description = "the sum of all entries"
    elif calc_type == "sum of absolute values":
        result = int(round(np.sum(np.abs(power_matrix))))
        calc_description = "the sum of absolute values of all entries"
    else:  # specific entry
        i, j = random.randint(0, rows-1), random.randint(0, rows-1)
        result = int(round(power_matrix[i, j]))
        calc_description = f"the value at position ({i+1}, {j+1})"
    
    matrix_str = format_matrix(matrix)
    
    templates = [
        f"Calculate the matrix power A^{power} and find {calc_description} of the result:\nA =\n{matrix_str}",
        f"Find {calc_description} of the {power}th power of the matrix:\n{matrix_str}",
        f"What is {calc_description} of the matrix A^{power} where A is:\n{matrix_str}?",
        f"For the matrix power A^{power} where A =\n{matrix_str}, determine {calc_description}."
    ]
    
    question = random.choice(templates)
    
    return Problem(question=question, answer=result)

# --- Main Function and CLI Interface ---

def main():
    """
    Command-line interface for generating matrix computation problems.
    """
    parser = argparse.ArgumentParser(description='Generate matrix computation problems.')
    parser.add_argument('--type', type=str, 
                      choices=['inverse', 'multiplication', 'exponential', 
                               'svd', 'determinant', 'rank', 'eigenvalues', 
                               'power', 'all'], 
                      default='all', help='Type of matrix problem to generate')
    parser.add_argument('--size', type=int, default=None, 
                      help='Size for square matrices or maximum dimension for non-square matrices')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows for the matrix')
    parser.add_argument('--cols', type=int, default=None, help='Number of columns for the matrix')
    parser.add_argument('--min_val', type=int, default=-5, help='Minimum value for matrix elements')
    parser.add_argument('--max_val', type=int, default=5, help='Maximum value for matrix elements')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--matrix_type', type=str, 
                      choices=['general', 'symmetric', 'diagonal', 'triangular', 'orthogonal'], 
                      default='general', help='Type of matrix to generate')
    parser.add_argument('--min_power', type=int, default=2, help='Minimum power for matrix power problems')
    parser.add_argument('--max_power', type=int, default=4, help='Maximum power for matrix power problems')
    parser.add_argument('--count', type=int, default=1, help='Number of problems to generate')
    parser.add_argument('--decimals', type=int, default=2, help='Number of decimal places to round to')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Set dimensions
    rows = args.rows if args.rows is not None else args.size
    cols = args.cols if args.cols is not None else args.size
    
    # Use default if neither is specified
    if rows is None:
        rows = 3
    if cols is None:
        cols = 3

    # Convert matrix_type to special_type
    special_type = None if args.matrix_type == 'general' else args.matrix_type

    if args.type == 'all':
        problem_types = ['inverse', 'multiplication', 'exponential', 'svd', 
                         'determinant', 'rank', 'eigenvalues', 'power']
    else:
        problem_types = [args.type]

    for _ in range(args.count):
        for problem_type in problem_types:
            print(f"\n--- {problem_type.upper()} PROBLEM ---")
            
            if problem_type == 'inverse':
                problem = generate_matrix_inverse_problem(rows, special_type, args.min_val, args.max_val)
            elif problem_type == 'multiplication':
                problem = generate_matrix_multiplication_problem(rows_a=rows, cols_a=rows, cols_b=rows, min_val=args.min_val, max_val=args.max_val)
            elif problem_type == 'exponential':
                problem = generate_matrix_exponential_problem(rows, special_type, args.min_val, args.max_val)
            elif problem_type == 'svd':
                problem = generate_svd_decomposition_problem(rows, cols, args.min_val, args.max_val)
            elif problem_type == 'determinant':
                problem = generate_matrix_determinant_problem(rows, special_type, args.min_val, args.max_val)
            elif problem_type == 'rank':
                problem = generate_matrix_rank_problem(rows, cols, args.min_val, args.max_val)
            elif problem_type == 'eigenvalues':
                problem = generate_eigenvalue_problem(rows, special_type, args.min_val, args.max_val)
            elif problem_type == 'power':
                problem = generate_matrix_power_problem(rows, special_type, args.min_val, args.max_val,
                                                       args.min_power, args.max_power)
            
            print(f"Question: {problem.question}")
            print(f"\nAnswer: {problem.answer}")
            print("-" * 50)


if __name__ == "__main__":
    main() 