"""
Number theory module containing various number theory problem generators.
This module generates problems related to prime mods, triple counts,
and digit sum with related number theory properties and constraints.

The module supports generating problems with different difficulty levels (1-5),
where higher levels require more reasoning steps and work usually work
with larger numbers and expect larger answers:
    - Level 1: Expecting one-digit to two-digit answers and 
        roughly working with two-digit numbers
    - Level 2: Expecting one-digit to two-digit answers and 
        roughly working with three-digit numbers
    - Level 3: Expecting two-digit to three-digit answers and 
        roughly working with four-digit numbers
    - Level 4: Expecting three-digit to five-digit answers and 
        roughly working with five-digit numbers
    - Level 5: Expecting four-digit (or more) answers and 
        roughly working with six-digits (or more) numbers

Each function generator returns a tuple of (function, expression, difficulty_info),
where difficulty_info is a dictionary containing metadata about the problem's complexity,
including other numbers that were not asked by the problem, but may provide valuable insight
into the problem's overall nuance.
"""

import math
import numpy as np
import random
from random import randint
import argparse
from sympy import primerange, nextprime, gcd
from functools import reduce
from collections import namedtuple


# Simple Problem container with question, answer, plot data, and difficulty information
Problem = namedtuple('Problem', ('question', 'answer', 'difficulty'))

# --- Function Generation Utilities ---

def find_prime_mod_1(first_n, first_p, offset, limit=1000):
    for p in primerange(2, limit):
        p2 = p ** first_p
        for n in range(1, p2):
            if (n ** first_n + offset) % p2 == 0:
                return p, n
    return None, None

def find_prime_mod_2(first_n, first_p, second_p, offset, loop_up, limit=1000):
    for p in primerange(2, limit):
        p2 = p ** first_p
        p3 = p ** second_p
        for n in range(1, p3): 
            if (n ** first_n + offset) % p2 == 0 and (n ** first_n + offset) % p3 == 0:
                start = 1 if loop_up else p3 - 1
                end = p3 if loop_up else 0
                step = 1 if loop_up else -1
                for m in range(start, end, step):
                    if (m ** first_n + offset) % p3 == 0:
                        return p, m
    return None, None

def find_prime_mod_3(n_pow, p_pow, offset):
    p = 2
    while True:
        p3 = p ** p_pow
        for n in range(p3):
            if (n ** n_pow + offset) % p3 == 0:
                return p, n
        p = nextprime(p)

def find_prime_mod_4(mod, n_pow, p_pow):
    p = 2
    while True:
        if p % mod != 1:
            p = nextprime(p)
            continue
        p2 = p ** p_pow
        for n in range(p2):
            if (n ** n_pow + 1) % p2 == 0:
                return p, n 
        p = nextprime(p)

def find_prime_mod_5(prime, power, m_pow):
    p2 = prime ** power
    solutions = []
    for m in range(1, p2 + 1):
        if (m ** m_pow + 1) % p2 == 0:
            solutions.append(m)
    return len(solutions), min(solutions) if solutions else None

# ------------

def find_triple_count_1(power, power2, base):
    limit = base ** power
    mod = base ** power2
    count = 0
    for a in range(1, limit + 1):
        if a ** base % mod == 0:
            count += 1
    return count % 1000

def find_triple_count_2(i, j, base, a_coef, b_coef):
    limit = base ** i
    mod = base ** j
    count = 0
    for a in range(1, limit + 1):
        for b in range(1, limit + 1):
            if (a_coef * a ** base + b_coef * b ** base) % mod == 0:
                count += 1
    return count % 1000

def find_triple_count_3(i, j, base, a_coef, b_coef, c_coef):
    limit = base ** i
    mod = base ** j
    count = 0
    for a in range(1, limit + 1):
        for b in range(1, limit + 1):
            for c in range(1, limit + 1):
                if (a_coef * a ** base + b_coef * b ** base + c_coef * c ** base) % mod == 0:
                    count += 1
    return count % 1000

def find_triple_count_4(i, j, base, a_coef, b_coef, c_coef, d_coef):
    limit = base ** i
    mod = base ** j
    count = 0
    for a in range(1, limit + 1):
        for b in range(1, limit + 1):
            for c in range(1, limit + 1):
                for d in range(1, limit + 1):
                    if (a_coef * a ** base + b_coef * b ** base + c_coef * c ** base + d_coef * d ** base) % mod == 0:
                        count += 1
    return count % 1000

def find_triple_count_5_coprime(i, j, base, a_coef, b_coef, c_coef):
    limit = base ** i
    mod = base ** j
    count = 0
    for a in range(1, limit + 1):
        for b in range(1, limit + 1):
            for c in range(1, limit + 1):
                if (a_coef * a ** base + b_coef * b ** base + c_coef * c ** base) % mod == 0 and gcd(gcd(a,  b), c) == 1:
                    count += 1
    return count % 1000

# ------------

def find_digit_sum(num_digits, target_digit, div, loop_back, mode):

    def is_valid_one_digit(n, target_digit, div):
        s = str(n)
        for i in range(len(s)):
            modified = list(s)
            modified[i] = str(target_digit)
            new_num = int(''.join(modified))
            if new_num % div != 0:
                return False
        return True

    def is_valid_two_digit(n, target_digit, div):
        s = str(n)
        for i in range(len(s) - 1):
            for j in range(i + 1, len(s)):
                modified = list(s)
                modified[i] = str(target_digit)
                modified[j] = str(target_digit)
                new_num = int(''.join(modified))
                if new_num % div != 0:
                    return False
        return True

    def is_valid_two_digit_swap(n, div):
        s = str(n)
        for i in range(1, len(s)):
            modified = list(s)
            modified[i - 1], modified[i] = modified[i], modified[i - 1]
            new_num = int(''.join(modified))
            if new_num % div != 0:
                return False
        return True

    def is_valid_alt_sum(n, div):
        s = str(n)
        tot = 0
        for i in range(len(s)):
            if i % 2 == 0:
                tot += int(s[i])
            else:
                tot -= int(s[i])
        if abs(tot) % div != 0:
            return False
        return True

    def is_valid_sum(n, div):
        s = str(n)
        tot = 0
        for i in range(len(s)):
            tot += int(s[i])
        if tot % div != 0:
            return False
        return True

    def is_valid_reverse(n, div):
        return n % div == 0 and int(str(n)[::-1]) % div == 0
    
    def find_lcm(a, b):
        temp = math.gcd(a, b)
        if temp == 0:
            return 0
        return abs(a * b) // temp

    def lcm_of_list(lst):
        return reduce(find_lcm, [int(n) for n in lst])
    
    cnt = 0
    start = int('9' * num_digits) if loop_back else int('9' * (num_digits - 1)) + 1
    end = int('9' * (num_digits - 1)) if loop_back else int('9' * num_digits) + 1
    step = -1 if loop_back else 1
    for n in range(start, end, step):
        if mode == 'one_digit_change':
            if is_valid_one_digit(n, target_digit, div):
                return n, n // int("10" + "0" * (num_digits - 2)), n % int("10" + "0" * (num_digits - 2))
        
        elif mode == 'two_digit_change':
            if is_valid_two_digit(n, target_digit, div):
                return n, n // int("10" + "0" * (num_digits - 2)), n % int("10" + "0" * (num_digits - 2))
        
        elif mode == 'two_digit_swap':
            if is_valid_two_digit_swap(n, div):
                cnt += 1
                if cnt == 3:
                    return n, n // int("10" + "0" * (num_digits - 2)), n % int("10" + "0" * (num_digits - 2))
                
        elif mode == 'alternating_sum':
            if is_valid_alt_sum(n, div):
                return n % int('10' + '0' * (num_digits - 2)), n // int("10" + "0" * (num_digits - 2)), n % int("10" + "0" * (num_digits - 2))
            
        elif mode == 'sum':
            if is_valid_sum(n, div):
                cnt += 1
                if cnt == 10:
                    return n, n // int("10" + "0" * (num_digits - 2)), n % int("10" + "0" * (num_digits - 2))
        
        elif mode == 'reverse':
            if is_valid_reverse(n, div):
                return n, n // int("10" + "0" * (num_digits - 2)), n % int("10" + "0" * (num_digits - 2))

        elif mode == 'one_digit_change_and_reverse':
            if is_valid_one_digit(n, target_digit, div):
                s = str(n)
                lcm = lcm_of_list(s)
                if lcm != 0:
                    n, q, r = find_digit_sum(num_digits + 1, 0, lcm, True, 'reverse')
                    if n:
                        return n, n // int("10" + "0" * (num_digits - 1)), n % int("10" + "0" * (num_digits - 1))
    
    return None, None, None

# --- Problem Generator Functions ---

def generate_prime_mod_problem(difficulty=3):
    """
    Generate a problem about finding multiples of prime powers
    
    Args:
        difficulty: Integer specifying the difficulty of the
        problems to be generated
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    if difficulty <= 3:
        first_n = randint(1, 5)
        first_p = randint(1, 5)
        second_p = randint(first_p, 5)
        offset = randint(-2, 2)
        n_pow = randint(1, 6)
        p_pow = randint(2, 4)
        mod = randint(2, 9)
        prime = random.sample(list(primerange(1, 40)), 1)[0]
        power = randint(1, 4)
        m_pow = randint(1, 4)
    else:
        first_n = randint(2, 6)
        first_p = randint(2, 6)
        second_p = randint(first_p, 6)
        offset = randint(-2, 2)
        n_pow = randint(2, 7)
        p_pow = randint(2, 4)
        mod = randint(2, 9)
        prime = random.sample(list(primerange(1, 60)), 1)[0]
        power = randint(2, 5)
        m_pow = randint(2, 5)
        
    templates = [
        f"Let p be the least prime number for which there exists a positive integer n such that n^{first_n}+({offset}) is divisible by p^{first_p}. Find the least positive integer m such that m^{first_n}+({offset}) is divisible by p^{first_p}.",
        f"Let p be the least prime number for which there exists a positive integer n such that n^{first_n}+({offset}) is divisible by p^{first_p}, and that n^{first_n}+({offset}) is also divisible by p^{second_p}. Find the smallest positive integer m such that m^{first_n}+({offset}) is divisible by p^{first_p}.",
        f"Let p be the smallest prime number for which there exists a positive integer n where n^{first_n}+({offset}) is a multiple of p^{first_p}, and n^{first_n}+({offset}) is also a multiple of p^{second_p}. Find the largest positive integer m such that m^{first_n}+({offset}) is a multiple of p^{first_p}.",
        f"Let p be the minimum prime where n^{n_pow}+{offset}=0 (mod p^{p_pow}) has a solution for some integer n, then find the minimum n for this p.",
        f"Let p be a prime with p=1 (mod {mod}). For the least p where, for some integer n, n^{n_pow}=-1 (mod p^{p_pow}) has a solution, find the minimal n for this p.",
        f"let p={prime}. How many solutions m in [1, 2, ..., {prime}^{power}] satisfy m^{m_pow}+1=0 (mod {prime}^{power})?",
        f"let p={prime}. Let N be the number of solutions m in [1, 2, ..., {prime}^{power}] that satisfy m^{m_pow}+1=0 (mod {prime}^{power}). Let M be the minimum of these solutions. What is N+M?"
    ]

    template_index = randint(0, len(templates) - 1)
    
    question = templates[template_index]

    if template_index == 0:
        p, m = find_prime_mod_1(first_n, first_p, offset)
        answer = m
    elif template_index == 1:
        p, m = find_prime_mod_2(first_n, first_p, second_p, offset, True)
        answer = m
    elif template_index == 2:
        p, m = find_prime_mod_2(first_n, first_p, second_p, offset, False)
        answer = m
    elif template_index == 3:
        p, n = find_prime_mod_3(n_pow, p_pow, offset)
        answer = n
    elif template_index == 4:
        p, n = find_prime_mod_4(mod, n_pow, p_pow)
        answer = n
    elif template_index == 5:
        length, minimum = find_prime_mod_5(prime, power, m_pow)
        answer = length
    elif template_index == 6:
        length, minimum = find_prime_mod_5(prime, power, m_pow)
        answer = length + minimum if length and minimum else None
    
    if template_index <= 4:
        difficulty_info = {
            "difficulty": difficulty,
            "p": p,
            "num": answer,
        }
    else:
        difficulty_info = {
            "difficulty": difficulty,
            "length": length,
            "min_sol": minimum
        }

    return Problem(question = question, answer = answer, difficulty = difficulty_info)

def generate_triple_count_problem(difficulty=3):
    """
    Generate a problem about finding number of ordered combinations 
    satisfying certain constraints
    
    Args:
        difficulty: Integer specifying the difficulty of the
        problems to be generated
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    if difficulty == 1:
        base = randint(2, 3)
        power = randint(2, 3)
        power2 = randint(2, 3)
        a_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        b_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        
        templates = [
            f"Assume N is the number of integers a which are >= 0 where a <= {base}^{power} and a^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Let N be the number of integers a which are non-negative where a <= {base}^{power} and a^{base} is a multiple of {base}^{power2}. Find N mod 1000.",
            f"Let N denote the number of ordered pairs of positive integers (a, b) such that a, b <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} is divisible by {base}^{power2}. Find N mod 1000."
        ]

        template_index = randint(0, len(templates) - 1)

        question = templates[template_index]

        if template_index == 0 or template_index == 1:
            answer = find_triple_count_1(power, power2, base)
        elif template_index == 2:
            answer = find_triple_count_2(power, power2, base, a_coef, b_coef)
            difficulty = 2 - (answer < 10) * 1

        difficulty_info = {
            "difficulty": difficulty,
            "num_variables": 1 if template_index == 0 or template_index == 1 else 2
        }

        return Problem(question = question, answer = answer, difficulty = difficulty_info)

    elif difficulty == 2:
        base = randint(3, 4)
        power = randint(3, 4)
        power2 = randint(2, 3)
        a_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        b_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        
        templates = [
            f"Assume N is the number of integers a which are >= 0 where a <= {base}^{power} and a^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Let N be the number of integers a which are non-negative where a <= {base}^{power} and a^{base} is a multiple of {base}^{power2}. Find N mod 1000.",
            f"Let N denote the number of ordered pairs of positive integers (a, b) such that a, b <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} is divisible by {base}^{power2}. Find N mod 1000."
        ]

        template_index = randint(0, len(templates) - 1)
        
        question = templates[template_index]
        
        if template_index == 0 or template_index == 1:
            answer = find_triple_count_1(power, power2, base)
            difficulty = 1 + (answer >= 10) * 1
        elif template_index == 2:
            answer = find_triple_count_2(power, power2, base, a_coef, b_coef)
        
        difficulty_info = {
            "difficulty": difficulty,
            "num_variables": 1 if template_index == 0 or template_index == 1 else 2
        }

        return Problem(question = question, answer = answer, difficulty = difficulty_info)
    
    elif difficulty == 3:
        base = randint(3, 4)
        power = randint(2, 3)
        power2 = randint(2, 3)
        a_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        b_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        c_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        
        templates = [
            f"Let N denote the number of ordered pairs of positive integers (a, b) such that a, b <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Call N the number of ordered pairs of positive integers (a, b) such that a, b <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} is a multiple of {base}^{power2}. Find N mod 1000.",
            f"Define M to be the number of ordered triples of positive integers (a, b, c) where a, b, c <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} = 0 (mod {base}^{power2}). Find the remainder of M when divided by 1000.",
            f"Let M be the number of ordered triples of positive integers (a, b, c) where a, b, c <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} = 0 (mod {base}^{power2}). Find M mod 1000."
        ]

        template_index = randint(0, len(templates) - 1)
        
        question = templates[template_index]
        
        if template_index == 0 or template_index == 1:
            answer = find_triple_count_2(power, power2, base, a_coef, b_coef)
            difficulty = 2 + (answer >= 100) * 1
        elif template_index == 2 or template_index == 3:
            answer = find_triple_count_3(power, power2, base, a_coef, b_coef, c_coef)
        
        difficulty_info = {
            "difficulty": difficulty,
            "num_variables": 2 if template_index == 0 or template_index == 1 else 3
        }

        return Problem(question = question, answer = answer, difficulty = difficulty_info)
    
    elif difficulty == 4:
        base = randint(3, 4)
        power = randint(2, 3)
        power2 = randint(2, 3)
        a_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        b_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        c_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        d_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]

        templates = [
            f"Define M to be the number of ordered triples of positive integers (a, b, c) where a, b, c <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} = 0 (mod {base}^{power2}). Find the remainder of M when divided by 1000.",
            f"Let M be the number of ordered triples of positive integers (a, b, c) where a, b, c <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} = 0 (mod {base}^{power2}). Find M mod 1000.",
            f"Let N be the number of ordered quadruples of integers (a, b, c, d) such that a, b, c, d <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} + ({d_coef})d^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Define N to be the number of ordered quadruples of integers (a, b, c, d) such that a, b, c, d <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} + ({d_coef})d^{base} is divisible by {base}^{power2}. Find N mod 1000.",
        ]

        template_index = randint(0, len(templates) - 1)
        
        question = templates[template_index]
        
        if template_index == 0 or template_index == 1:
            answer = find_triple_count_3(power, power2, base, a_coef, b_coef, c_coef)
            difficulty = 3 + (answer >= 100) * 1
        elif template_index == 2 or template_index == 3:
            answer = find_triple_count_4(power, power2, base, a_coef, b_coef, c_coef, d_coef)

        difficulty_info = {
            "difficulty": difficulty,
            "num_variables": 3 if template_index == 0 or template_index == 1 else 4
        }
            
        return Problem(question = question, answer = answer, difficulty = difficulty_info)
    
    else: # difficulty 5
        base = randint(3, 4)
        power = randint(2, 3)
        power2 = randint(2, 3)
        a_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        b_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        c_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        d_coef = randint(1, 4) * random.sample((-1, 1), 1)[0]
        aa_coef = randint(-1, 1)
        bb_coef = randint(-1, 1)
        cc_coef = randint(-1, 1)

        templates = [
            f"Let N be the number of ordered quadruples of integers (a, b, c, d) such that a, b, c, d <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} + ({d_coef})d^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Define N to be the number of ordered quadruples of integers (a, b, c, d) such that a, b, c, d <= {base}^{power} and ({a_coef})a^{base} + ({b_coef})b^{base} + ({c_coef})c^{base} + ({d_coef})d^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Find the number of ordered triples T of positive integers (a, b, c) satisfying three conditions: 1. a, b, c <= {base}^{power}, 2. a, b, c are relatively coprime, and 3. ({aa_coef})a^{base} + ({bb_coef})b^{base} + ({cc_coef})c^{base} is divisible by {base}^{power2}. Find N mod 1000.",
            f"Find the number of ordered triples T of positive integers (a, b, c) satisfying three conditions: 1. a, b, c <= {base}^{power}, 2. a, b, c are relatively coprime, and 3. ({aa_coef})a^{base} + ({bb_coef})b^{base} + ({cc_coef})c^{base} is a multiple of {base}^{power2}. Find N mod 1000."
        ]

        template_index = randint(0, len(templates) - 1)
        
        question = templates[template_index]
        
        if template_index == 0 or template_index == 1:
            answer = find_triple_count_4(power, power2, base, a_coef, b_coef, c_coef, d_coef)
            difficulty = 4 + (answer >= 100) * 1
        elif template_index == 2 or template_index == 3:
            answer = find_triple_count_5_coprime(power, power2, base, aa_coef, bb_coef, cc_coef)

        difficulty_info = {
            "difficulty": difficulty,
            "num_variables": 4 if template_index == 0 or template_index == 1 else 3
        }
            
        return Problem(question = question, answer = answer, difficulty = difficulty_info)

def generate_digit_sum_problem(difficulty=3):
    """
    Generate a problem about finding the largest/smallest
    number with its digits satisfying certain properties
    
    Args:
        difficulty: Integer specifying the difficulty of the
        problems to be generated
        
    Returns:
        Problem: A Problem object with question, answer, and difficulty
    """
    if difficulty == 1:
        num_digits = randint(1, 2)
        div = randint(3, 7)
    elif difficulty == 5:
        num_digits = randint(6, 7)
        div = randint(5, 12)
    else:
        num_digits = difficulty + 1
        div = randint(3, 9)
        
    target_digit = randint(1, 9)
    
    templates = [
        f"let N be the maximum {num_digits}-digit positive integer with the property that whenever one of its digits changes to {target_digit}, the resulting number is a multiple of {div}. Let Q and R be the quotient and remainder, respectively, when N is divided by {int('10' + '0' * (num_digits - 2))}. Find Q + R.",
        f"let N be the smallest {num_digits}-digit positive integer with the property that whenever one of its digits changes to {target_digit}, the resulting number is a multiple of {div}. Let Q and R be the quotient and remainder, respectively, when N is divided by {int('10' + '0' * (num_digits - 2))}. Find Q + R.",
        f"let N be the biggest {num_digits}-digit positive integer with the property that whenever two of its digits changes to {target_digit}, the resulting number is a multiple of {div}. Let Q and R be the quotient and remainder, respectively, when N is divided by {int('10' + '0' * (num_digits - 2))}. Find Q + R.",
        f"let N be the least {num_digits}-digit positive integer with the property that whenever two of its digits changes to {target_digit}, the resulting number is a multiple of {div}. Let Q and R be the quotient and remainder, respectively, when N is divided by {int('10' + '0' * (num_digits - 2))}. Find Q + R.",
        f"let N be the third greatest {num_digits}-digit positive integer with the property that whenever you swap the positions of two of its adjacent digits, the resulting number is a multiple of {div}. Find N.",
        f"let N be the largest {num_digits}-digit positive integer with the property that the absolute value of the alternating sum (starting with adding the most significant digit) of the digits of n is divisible by {div}. Find N mod {int('10' + '0' * (num_digits - 2))}.",
        f"let N be the tenth greatest {num_digits}-digit positive integer with the property that the sum of the digits of n is divisible by {div}. Find N.",
        f"let N be the third smallest {num_digits}-digit positive integer with the property that whenever you swap the positions of two of its adjacent digits, the resulting number is a multiple of {div}. Find N.",
        f"let N be the minimum {num_digits}-digit positive integer with the property that the absolute value of the alternating sum (starting with adding the most significant digit) of the digits of n is divisible by {div}. Find N mod {int('10' + '0' * (num_digits - 2))}.",
        f"let N be the tenth smallest {num_digits}-digit positive integer with the property that the sum of the digits of n is divisible by {div}. Find N.",
        f"let N be the greatest {num_digits}-digit positive integer with the property that both the number and the number formed by reversing its digits are divisible by {div}. Let Q and R be the quotient and remainder, respectively, when N is divided by {int('10' + '0' * (num_digits - 2))}. Find Q + R.",
        f"let N be the least {num_digits}-digit positive integer with the property that both the number and the number formed by reversing its digits are divisible by {div}. Let Q and R be the quotient and remainder, respectively, when N is divided by {int('10' + '0' * (num_digits - 2))}. Find Q + R."
    ]

    if difficulty >= 4:
        templates.append(f"let N be the maximum {num_digits}-digit positive integer with the property that whenever one of its digits changes to {target_digit}, the resulting number is a multiple of {div}. Let S be the Lowest Common Multiple (LCM) of the digits of N. let M be the largest {num_digits + 1}-digit positive integer with the property that both the number and the number formed by reversing its digits are divisible by S. Let Q and R be the quotient and remainder, respectively, when M is divided by {int('10' + '0' * (num_digits - 1))}. Find Q + R.")

    template_index = randint(0, len(templates) - 1)
    
    question = templates[template_index]

    if template_index == 0:
        is_find_largest = True
        problem_type = 'one_digit_change'
    elif template_index == 1:
        is_find_largest = False
        problem_type = 'one_digit_change'
    elif template_index == 2:
        is_find_largest = True
        problem_type = 'two_digit_change'
    elif template_index == 3:
        is_find_largest = False
        problem_type = 'two_digit_change'
    elif template_index == 4:
        is_find_largest = True
        problem_type = 'two_digit_swap'
    elif template_index == 5:
        is_find_largest = True
        problem_type = 'alternating_sum'
    elif template_index == 6:
        is_find_largest = True
        problem_type = 'sum'
    elif template_index == 7:
        is_find_largest = False
        problem_type = 'two_digit_swap'
    elif template_index == 8:
        is_find_largest = False
        problem_type = 'alternating_sum'
    elif template_index == 9:
        is_find_largest = False
        problem_type = 'sum'
    elif template_index == 10:
        is_find_largest = True
        problem_type = 'reverse'
    elif template_index == 11:
        is_find_largest = False
        problem_type = 'reverse'
    elif template_index == 12:
        is_find_largest = False
        problem_type = 'one_digit_change_and_reverse'
    
    n, q, r = find_digit_sum(num_digits, target_digit, div, is_find_largest, problem_type)

    if not n or not q or not r:
        answer = None
    elif template_index <= 3:
        answer = q + r
    elif template_index >= 4:
        answer = n

    if template_index == 5 or template_index == 8:
        difficulty_info = {
            "difficulty": difficulty,
            "N mod": n,
            "Q": q,
            "R": r
        }
    else:
        difficulty_info = {
            "difficulty": difficulty,
            "N": n,
            "Q": q,
            "R": r
        }

    return Problem(question = question, answer = answer, difficulty = difficulty_info)

# --- Main Function and CLI Interface ---

def main():
    """
    Command-line interface for generating number theory problems.
    """
    parser = argparse.ArgumentParser(description='Generate number theory problems.')
    parser.add_argument('--type', type=str,
                      choices=['prime_mod, triple_count, digit_sum'], 
                      default='all', help='Type of function problem to generate')
    parser.add_argument('--count', type=int, default=10, help='Number of problems to generate')
    parser.add_argument('--difficulty', type=int, choices=[1, 2, 3, 4, 5], default=None, 
                     help='Difficulty level (1-5, where 5 is hardest)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.type == 'all':
        problem_types = ['prime_mod', 'triple_count', 'digit_sum']
    else:
        problem_types = [args.type]

    for _ in range(args.count):
        for problem_type in problem_types:
            print(f"\n--- {problem_type.upper()} PROBLEM ---")
            
            if problem_type == 'prime_mod':
                problem = generate_prime_mod_problem(args.difficulty)
            elif problem_type == 'triple_count':
                problem = generate_triple_count_problem(args.difficulty)
            elif problem_type == 'digit_sum':
                problem = generate_digit_sum_problem(args.difficulty)
            
            print(f"Question: {problem.question}")
            print(f"\nAnswer: {problem.answer}")
            
            # Print difficulty information if available
            if hasattr(problem, 'difficulty') and problem.difficulty:
                print(f"\nDifficulty Level: {problem.difficulty.get('level', 'Unknown')}")
            
            print("-" * 50)

if __name__ == "__main__":
    main() 
    