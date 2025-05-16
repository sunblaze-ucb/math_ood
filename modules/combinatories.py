"""
Combinatories module containing letter distribution and probability problem generators.
This is a merged module combining functionality from:
- combinatories_v2.py
- combinatories_distribute_letter.py
"""

import random
import string
import math
import argparse
import json
import re
from itertools import product, permutations, combinations
from collections import Counter
from collections import namedtuple

# Simple Problem container (from example.py)
Problem = namedtuple('Problem', ('question', 'answer'))

# --- Common Utility Functions ---

def partition_integer(n, parts, min_val=1, max_val=None):
    """
    Partition integer n into exactly 'parts' positive integers.

    Args:
        n: Integer to partition
        parts: Number of parts
        min_val: Minimum value for each part
        max_val: Maximum value for each part

    Returns:
        list: A list of integers that sum to n
    """
    if max_val is None:
        max_val = n - (parts - 1) * min_val

    if parts <= 0 or n < parts * min_val or (max_val is not None and n > parts * max_val):
        raise ValueError("Cannot partition with these constraints")

    # Initialize with minimum values
    result = [min_val] * parts
    remaining = n - sum(result)

    # Distribute remaining amount randomly
    while remaining > 0:
        idx = random.randint(0, parts - 1)
        if result[idx] < max_val:
            result[idx] += 1
            remaining -= 1

    return result


def reduce_fraction(numerator, denominator):
    """
    Reduce a fraction to its simplest form.
    Returns a tuple (numerator, denominator).
    """
    # print(numerator, denominator)
    if numerator == 0:
        return (0, 1)
    elif numerator == denominator:
        return (1, 1)
    
    # Reduce the fraction
    gcd = math.gcd(numerator, denominator)
    
    return (numerator // gcd, denominator // gcd)


# --- Permutation Probability Functions (from combinatories_v2.py) ---

def is_valid_word(word, max_counts):
    """
    Check if a word is valid given the maximum counts for each letter.
    """
    counter = Counter(word)
    for letter, count in counter.items():
        if count > max_counts.get(letter, 0):
            return False
    return True


def generate_all_words(length, max_counts):
    """
    Generate all possible words of given length from the multiset, along with their
    sampling probabilities as (numerator, denominator) pairs.
    """
    letters = list(max_counts.keys())
    all_words = {}

    # Calculate total ways to select letters from the multiset
    total_letters = sum(max_counts.values())
    denominator = 1
    for i in range(length):
        denominator *= (total_letters - i)

    for p in product(letters, repeat=length):
        word = ''.join(p)
        if is_valid_word(word, max_counts):
            # Calculate numerator: ways to select these specific letters
            counter = Counter(word)
            numerator = 1
            remaining_counts = max_counts.copy()
            
            for letter in word:
                numerator *= remaining_counts[letter]
                remaining_counts[letter] -= 1
                
            all_words[word] = (numerator, denominator)

    return all_words


def count_fixed_points(original, permuted):
    """
    Count how many letters remain in their original positions.
    """
    return sum(a == b for a, b in zip(original, permuted))


def count_fixed_points_for_letter(original, permuted, letter):
    """
    Count how many of a specific letter remain in their original positions.
    """
    return sum(a == b and a == letter for a, b in zip(original, permuted))


def shift_word(word, shift):
    """
    Shift the letters in a word by a specified number of positions.
    """
    return word[-shift:] + word[:-shift]


# Event templates and calculators
EVENT_DESCRIPTIONS = {
    'no_fixed_points': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of no same letter remains in its original position?",
    'no_specific_letter_fixed': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of no letter '{special_letter}' occupy any of their original positions?",
    'exactly_n_specific_fixed': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of exact {n_specific_letters} letter '{special_letter}' remains in the same position?",
    'at_least_n_specific_fixed': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of at least {n_specific_letters} letter '{special_letter}' remains in the same position?",
    'exactly_n_after_shift_m': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shift the letter by {shift} position {example}, what is the probability of exact {n_specific_letters} letter '{special_letter}' remains in the same position?",
    'at_least_n_after_shift_m': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shift the letter by {shift} position {example}, what is the probability of at least {n_specific_letters} letter '{special_letter}' remains in the same position?",

    # 'exactly_n_after_shift_m': "Randomly select a word with {length} letters from the multiset {{{letters_str}}} (where each word has equal probability of being selected), shift the letter by {shift} position {example}, what is the probability of exact {n_specific_letters} letter '{special_letter}' remains in its original position?",
    # 'at_least_n_after_shift_m': "Randomly select a word with {length} letters from the multiset {{{letters_str}}} (where each word has equal probability of being selected), shift the letter by {shift} position {example}, what is the probability of at least {n_specific_letters} letter '{special_letter}' remains in its original position?",
}

EVENT_CALCULATORS = {
    'no_fixed_points': lambda words, **kwargs: probability_no_letter_in_original_position(words),
    'no_specific_letter_fixed': lambda words, **kwargs: probability_no_specific_letter_in_original_position(words, kwargs['special_letter']),
    'exactly_n_specific_fixed': lambda words, **kwargs: probability_exact_n_specific_letter_in_original_position(words,
                                                                                          kwargs['special_letter'],
                                                                                          kwargs['n_specific_letters']),
    'at_least_n_specific_fixed': lambda words, **kwargs: probability_at_least_n_specific_letter_in_original_position(words,
                                                                                          kwargs['special_letter'],
                                                                                          kwargs['n_specific_letters']),
    'exactly_n_after_shift_m': lambda words, **kwargs: probability_exact_n_specific_letter_after_shift(words, kwargs['special_letter'],
                                                                                 kwargs['n_specific_letters'],
                                                                                 kwargs['shift']),
    'at_least_n_after_shift_m': lambda words, **kwargs: probability_at_least_n_specific_letter_after_shift(words, kwargs['special_letter'],
                                                                                 kwargs['n_specific_letters'],
                                                                                 kwargs['shift']),
}


# --- Event-specific probability calculator functions ---

def probability_no_letter_in_original_position(words):
    """
    Calculate the probability that no letter remains in its original position
    after shuffling a randomly selected word.
    Returns a tuple (numerator, denominator) as a reduced fraction.
    """
    count_success = 0
    total_events = 0

    for word, (word_num, word_denom) in words.items():
        word_success = 0
        word_total = 0
        
        for perm in permutations(word):
            permuted_word = ''.join(perm)
            word_total += 1

            if count_fixed_points(word, permuted_word) == 0:
                word_success += 1
        
        # Scale by the probability of sampling this word
        count_success += word_success * word_num
        total_events = word_total * word_denom

    return reduce_fraction(count_success, total_events) if total_events > 0 else (0, 0)


def probability_no_specific_letter_in_original_position(words, special_letter):
    """
    Calculate the probability that no letter of a specific type remains in its original position
    after shuffling a randomly selected word.
    Returns a tuple (numerator, denominator) as a reduced fraction.
    """
    count_success = 0
    total_events = 0

    for word, (word_num, word_denom) in words.items():
        word_success = 0
        word_total = 0
        
        for perm in permutations(word):
            permuted_word = ''.join(perm)
            word_total += 1

            if count_fixed_points_for_letter(word, permuted_word, special_letter) == 0:
                word_success += 1
        
        # Scale by the probability of sampling this word
        count_success += word_success * word_num
        total_events = word_total * word_denom

    return reduce_fraction(count_success, total_events) if total_events > 0 else (0, 0)


def probability_exact_n_specific_letter_in_original_position(words, special_letter, n):
    """
    Calculate the probability that exactly n letters of a specific type remain in their original positions
    after shuffling a randomly selected word.
    Returns a tuple (numerator, denominator) as a reduced fraction.
    """
    count_success = 0
    total_events = 0

    for word, (word_num, word_denom) in words.items():
        word_success = 0
        word_total = 0
        
        for perm in permutations(word):
            permuted_word = ''.join(perm)
            word_total += 1

            if count_fixed_points_for_letter(word, permuted_word, special_letter) == n:
                word_success += 1
        
        # Scale by the probability of sampling this word
        count_success += word_success * word_num
        total_events = word_total * word_denom

    return reduce_fraction(count_success, total_events) if total_events > 0 else (0, 0)


def probability_at_least_n_specific_letter_in_original_position(words, special_letter, n):
    """
    Calculate the probability that at least n letters of a specific type remain in their original positions
    after shuffling a randomly selected word.
    Returns a tuple (numerator, denominator) as a reduced fraction.
    """
    count_success = 0
    total_events = 0

    for word, (word_num, word_denom) in words.items():
        word_success = 0
        word_total = 0
        
        for perm in permutations(word):
            permuted_word = ''.join(perm)
            word_total += 1

            if count_fixed_points_for_letter(word, permuted_word, special_letter) >= n:
                word_success += 1
        
        # Scale by the probability of sampling this word
        count_success += word_success * word_num
        total_events = word_total * word_denom

    return reduce_fraction(count_success, total_events) if total_events > 0 else (0, 0)


def probability_exact_n_specific_letter_after_shift(words, special_letter, n, shift):
    """
    Calculate the probability that exactly n letters of a specific type remain in their original positions
    after shifting a randomly selected word by the specified number of positions.
    Returns a tuple (numerator, denominator) as a reduced fraction.
    """
    count_success = 0
    total_weight = 0

    for word, (word_num, word_denom) in words.items():
        shifted_word = shift_word(word, shift)

        if count_fixed_points_for_letter(word, shifted_word, special_letter) == n:
            count_success += word_num
        
        total_weight += word_num

    return reduce_fraction(count_success, total_weight) if total_weight > 0 else (0, 0)


def probability_at_least_n_specific_letter_after_shift(words, special_letter, n, shift):
    """
    Calculate the probability that at least n letters of a specific type remain in their original positions
    after shifting a randomly selected word by the specified number of positions.
    Returns a tuple (numerator, denominator) as a reduced fraction.
    """
    count_success = 0
    total_weight = 0

    for word, (word_num, word_denom) in words.items():
        shifted_word = shift_word(word, shift)

        if count_fixed_points_for_letter(word, shifted_word, special_letter) >= n:
            count_success += word_num
        
        total_weight += word_num

    return reduce_fraction(count_success, total_weight) if total_weight > 0 else (0, 0)


def generate_probability_problem(event_type=None, length=None, letters=None, num_letters=None,
                     special_letter=None, n_specific_letters=None, shift=None,
                     random_seed=None, total_letters=None):
    """
    Generate a probability problem with specified or random parameters.

    Args:
        event_type: None for random
        length: Word length or None for random
        letters: List of letters to use or None for random
        letter_counts: Dict of letter counts or None for random
        special_letter: Special letter for events b-e or None for random
        n_specific_letters: Number of specific letters for events c-e or None for random
        shift: Shift amount for events d-e or None for random
        random_seed: Random seed for reproducibility or None
        total_letters: Total number of letters across all letter types

    Returns:
        tuple: (problem_statement, words, probability)
    """
    if random_seed is not None:
        random.seed(random_seed)

    # If event_type is None, choose randomly
    all_event_types = ['no_fixed_points', 'no_specific_letter_fixed', 'exactly_n_specific_fixed', 
                       'at_least_n_specific_fixed', 'exactly_n_after_shift_m', 'at_least_n_after_shift_m']
    
    if event_type is None:
        event_type = random.choice(all_event_types)

    # Generate or use provided length
    if length is None:
        length = random.randint(3, 4)
        
    # Set default total_letters if not provided
    if total_letters is None:
        total_letters = random.randint(5, 10)
        
    # Generate or use provided letters and counts
    if letters is None:
        available_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        if num_letters is None:
            num_letters = random.randint(2, 4)
        letters = random.sample(available_letters, num_letters)

        # Generate counts that sum to the total
        counts = partition_integer(total_letters, len(letters), 1)
        letter_counts = dict(zip(letters, counts))

    # Generate or use provided special letter (for events b-e)
    events_needing_special_letter = [
        'no_specific_letter_fixed', 'exactly_n_specific_fixed', 'at_least_n_specific_fixed', 
        'exactly_n_after_shift_m', 'at_least_n_after_shift_m'
    ]
    
    if event_type in events_needing_special_letter and special_letter is None:
        special_letter = random.choice(list(letter_counts.keys()))

    # Generate or use provided n_specific_letters (for events c-e)
    events_needing_n_specific = [
        'exactly_n_specific_fixed', 'at_least_n_specific_fixed', 
        'exactly_n_after_shift_m', 'at_least_n_after_shift_m'
    ]
    
    if event_type in events_needing_n_specific and n_specific_letters is None:
        n_specific_letters = random.randint(1, min(3, letter_counts.get(special_letter, 3)))

    # Generate or use provided shift (for events d-e)
    events_needing_shift = [
        'exactly_n_after_shift_m', 'at_least_n_after_shift_m'
    ]
    
    if event_type in events_needing_shift and shift is None:
        shift = random.randint(1, length - 1)

    # Generate all valid words
    words_dict = generate_all_words(length, letter_counts)
    
    # List of words for display
    words_list = list(words_dict.keys())

    # Prepare the problem context
    letters_str = ', '.join([f"{letter}: {count}" for letter, count in letter_counts.items()])

    # Format the specific event template
    if event_type == 'no_fixed_points':
        template = EVENT_DESCRIPTIONS[event_type].format(length=length, letters_str=letters_str)
    elif event_type == 'no_specific_letter_fixed':
        template = EVENT_DESCRIPTIONS[event_type].format(special_letter=special_letter, length=length, letters_str=letters_str)
    elif event_type == 'exactly_n_specific_fixed':
        template = EVENT_DESCRIPTIONS[event_type].format(special_letter=special_letter, n_specific_letters=n_specific_letters, length=length, letters_str=letters_str)
    elif event_type == 'at_least_n_specific_fixed':
        template = EVENT_DESCRIPTIONS[event_type].format(special_letter=special_letter, n_specific_letters=n_specific_letters, length=length, letters_str=letters_str)
    elif event_type == 'exactly_n_after_shift_m':
        # Create a dynamic example based on the shift value
        example_word = ''.join(chr(ord('a') + i) for i in range(5))  # "abcde"
        shifted_example = shift_word(example_word, shift)
        example_str = f" (eg. {example_word} -> {shifted_example})"
        template = EVENT_DESCRIPTIONS[event_type].format(
            special_letter=special_letter, 
            n_specific_letters=n_specific_letters, 
            shift=shift,
            example=example_str,
            length=length, 
            letters_str=letters_str
        )
    elif event_type == 'at_least_n_after_shift_m':
        # Create a dynamic example based on the shift value
        example_word = ''.join(chr(ord('a') + i) for i in range(5))  # "abcde"
        shifted_example = shift_word(example_word, shift)
        example_str = f" (eg. {example_word} -> {shifted_example})"
        template = EVENT_DESCRIPTIONS[event_type].format(
            special_letter=special_letter, 
            n_specific_letters=n_specific_letters, 
            shift=shift,
            example=example_str,
            length=length, 
            letters_str=letters_str
        )

    # Calculate the probability
    prob_kwargs = {
        'special_letter': special_letter,
        'n_specific_letters': n_specific_letters,
        'shift': shift
    }
    probability = EVENT_CALCULATORS[event_type](words_dict, **prob_kwargs)

    # For multiset sampling, context is included in the template
    problem_statement = f"What is the probability of such event happening: {template}\nIf the probability can be written as the form $\\frac{{m}}{{n}}$, where $m$ and $n$ are relatively prime integers, find $m + n$."

    return problem_statement, words_list, probability


# --- Letter Distribution Functions (from combinatories_distribute_letter.py) ---

def generate_letter_counts(num_letter_types, min_count=1, max_count=10, total_letters=None):
    """
    Generate a dictionary of letters and their counts.

    Args:
        num_letter_types: Number of different letter types to use
        min_count: Minimum count for each letter
        max_count: Maximum count for each letter
        total_letters: If provided, ensure the total number of letters equals this value

    Returns:
        dict: Dictionary mapping letters to their counts
    """
    # Select random letters
    letters = random.sample(string.ascii_lowercase, num_letter_types)

    if total_letters:
        # Generate counts that sum to the total
        counts = partition_integer(total_letters, num_letter_types, min_count)
    else:
        # Generate random counts
        counts = [random.randint(min_count, max_count) for _ in range(num_letter_types)]

    return dict(zip(letters, counts))


def generate_box_sizes(num_boxes, min_size=1, max_size=10, total_capacity=None):
    """
    Generate box sizes.

    Args:
        num_boxes: Number of boxes
        min_size: Minimum size of each box
        max_size: Maximum size of each box
        total_capacity: If provided, ensure total capacity equals total letters

    Returns:
        list: List of box sizes
    """
    if total_capacity:
        box_sizes = partition_integer(total_capacity, num_boxes, min_size, max_size)
    else:
        box_sizes = [random.randint(min_size, max_size) for _ in range(num_boxes)]

    return box_sizes


def generate_distribution_problem(num_letter_types=None, num_boxes=None, total_letters=None, labeled_boxes=False):
    """
    Generate a complete letter distribution problem.

    Args:
        num_letter_types: Number of different letter types
        num_boxes: Number of boxes
        total_letters: Total number of letters
        labeled_boxes: Whether boxes are labeled or not

    Returns:
        dict: Problem specification including letters, counts, and box sizes
    """
    if num_letter_types is None:
        num_letter_types = random.randint(2, 5)

    if num_boxes is None:
        num_boxes = random.randint(2, 6)
        
    if total_letters is None:
        total_letters = random.randint(6, 15)

    # Generate letter counts
    letters = generate_letter_counts(num_letter_types, total_letters=total_letters)

    # Generate box sizes that can fit all letters
    box_sizes = generate_box_sizes(num_boxes, total_capacity=total_letters)

    return {
        "letters": letters,
        "box_sizes": box_sizes,
        "labeled_boxes": labeled_boxes,
        "problem_type": "labeled" if labeled_boxes else "unlabeled"
    }


def find_letter_distributions(letters, box_sizes, labeled_boxes=False):
    """
    Find all unique ways to distribute letters into boxes of specific sizes.

    Args:
        letters (dict): Dictionary mapping letters to their counts
        box_sizes (list): List of box sizes (number of letters in each box)
        labeled_boxes (bool): If True, treat boxes as labeled (box 1 is different from box 2)
                             If False, treat boxes as unlabeled (only contents matter)

    Returns:
        list: List of all unique distributions
        int: Number of unique distributions
    """
    # Validate inputs
    total_letters = sum(letters.values())
    total_capacity = sum(box_sizes)

    if total_letters != total_capacity:
        raise ValueError(f"Total letters ({total_letters}) must equal sum of box sizes ({total_capacity})")

    # Convert the letter counts to a flat list of letters
    letter_list = []
    for letter, count in letters.items():
        letter_list.extend([letter] * count)

    # Group box sizes by their values for canonical representation (only used if labeled_boxes=False)
    box_size_groups = {}
    for i, size in enumerate(box_sizes):
        if size not in box_size_groups:
            box_size_groups[size] = []
        box_size_groups[size].append(i)

    # Store unique distributions and their representations
    unique_distributions = set()
    all_distributions = []

    # Recursive function to distribute letters into boxes
    def distribute_recursively(remaining_positions, box_index=0, current_boxes=None):
        if current_boxes is None:
            current_boxes = [[] for _ in range(len(box_sizes))]

        # Base case: all boxes are filled except the last one
        if box_index == len(box_sizes) - 1:
            # Put all remaining letters in the last box
            last_box = [letter_list[i] for i in remaining_positions]

            # Check if the last box has the correct size
            if len(last_box) == box_sizes[box_index]:
                # Complete the current distribution
                final_boxes = current_boxes.copy()
                final_boxes[box_index] = last_box

                # Convert to a canonical representation based on the labeled_boxes option
                if labeled_boxes:
                    canonical, readable = get_labeled_representation(final_boxes)
                else:
                    canonical, readable = get_unlabeled_representation(final_boxes, box_size_groups)

                if canonical not in unique_distributions:
                    unique_distributions.add(canonical)
                    all_distributions.append(readable)
            return

        # Try all ways to select positions for the current box
        for positions in combinations(remaining_positions, box_sizes[box_index]):
            # Create the current box
            current_box = [letter_list[i] for i in positions]

            # Update boxes
            new_boxes = current_boxes.copy()
            new_boxes[box_index] = current_box

            # Calculate remaining positions
            new_remaining = [pos for pos in remaining_positions if pos not in positions]

            # Move to the next box
            distribute_recursively(new_remaining, box_index + 1, new_boxes)

    def get_unlabeled_representation(boxes, box_size_groups):
        """Convert boxes to a canonical representation treating boxes as unlabeled"""
        # Convert each box to a counter and a readable string
        box_counters = []
        readable_boxes = []

        for i, box in enumerate(boxes):
            counter = Counter(box)
            # Convert counter to a sorted tuple of (letter, count) pairs
            counter_tuple = tuple(sorted((letter, count) for letter, count in counter.items()))
            # Store with box size for grouping
            box_counters.append((box_sizes[i], counter_tuple))

            # Create readable representation
            box_str = "".join(f"{letter}:{count}" for letter, count in sorted(counter.items()))
            readable_boxes.append(f"Box {i + 1}({box_sizes[i]}): {box_str}")

        # Group boxes by size
        size_groups = {}
        for size, counter in box_counters:
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(counter)

        # Sort boxes within each size group
        canonical_parts = []
        for size in sorted(size_groups.keys()):
            sorted_group = sorted(size_groups[size])
            canonical_parts.append((size, tuple(sorted_group)))

        return tuple(canonical_parts), " | ".join(readable_boxes)

    def get_labeled_representation(boxes):
        """Convert boxes to a representation treating boxes as labeled"""
        # Convert each box to a counter and a readable string
        canonical_parts = []
        readable_boxes = []

        for i, box in enumerate(boxes):
            counter = Counter(box)
            # Convert counter to a sorted tuple of (letter, count) pairs
            counter_tuple = tuple(sorted((letter, count) for letter, count in counter.items()))
            # Add to canonical parts with box index to maintain box identity
            canonical_parts.append((i, counter_tuple))

            # Create readable representation
            box_str = "".join(f"{letter}:{count}" for letter, count in sorted(counter.items()))
            readable_boxes.append(f"Box {i + 1}({box_sizes[i]}): {box_str}")

        return tuple(canonical_parts), " | ".join(readable_boxes)

    # Start the distribution process
    distribute_recursively(list(range(total_letters)))

    return all_distributions, len(unique_distributions)


def distribute_letter_problem(num_letter_type_range=None, num_boxes_range=None, total_letters_range=None):
    """
    Generate a letter distribution problem and calculate all possible distributions.

    Args:
        num_letter_type_range: Range of number of different letter types
        num_boxes_range: Range of number of boxes
        total_letters_range: Range of total letters

    Returns:
        Problem: A Problem object with question and answer
    """
    # Default ranges if not provided
    if num_letter_type_range is None:
        num_letter_type_range = [2, 3, 4]
    if num_boxes_range is None:
        num_boxes_range = [2, 3, 4]
    if total_letters_range is None:
        total_letters_range = [6, 8, 10, 12, 14, 16]

    labeled_boxes = True if random.random() < 0.5 else False
    num_letter_types = random.choice(num_letter_type_range)
    num_boxes = random.choice(num_boxes_range)
    total_letters = random.choice(total_letters_range)
    problem = generate_distribution_problem(num_letter_types, num_boxes, total_letters, labeled_boxes)

    # Find all distributions
    distributions, count = find_letter_distributions(
        problem["letters"],
        problem["box_sizes"],
        problem["labeled_boxes"]
    )

    labeled_templates = [
        "Divide the letters from {letters} into {num_boxes} distinctively labeled boxes with sizes {box_sizes}. How many ways?",
        "You have {letters} and need to distribute them into {num_boxes} different boxes with capacities {box_sizes}. In how many ways can this be done?",
        "Given the letters {letters}, place them into {num_boxes} labeled boxes with sizes {box_sizes}. How many distinct arrangements are possible?",
        "Distribute the letters {letters} among {num_boxes} distinguishable boxes with capacities {box_sizes}. How many different distributions exist?",
        "Arrange the letters {letters} into {num_boxes} uniquely identified boxes that can hold {box_sizes} letters respectively. Find the number of possible arrangements."
    ]
    
    unlabeled_templates = [
        "Divide the letters from {letters} into {num_boxes} undistinguishable boxes with sizes {box_sizes}. How many ways?",
        "You have {letters} and need to distribute them into {num_boxes} identical boxes with capacities {box_sizes}. How many distinct distributions are possible?",
        "Given the letters {letters}, place them into {num_boxes} indistinguishable boxes with sizes {box_sizes}. In how many ways can you do this?",
        "Distribute the letters {letters} among {num_boxes} identical containers that can hold {box_sizes} letters respectively. How many different arrangements exist?",
        "Arrange the letters {letters} into {num_boxes} undistinguishable boxes with capacities {box_sizes}. Find the total number of possible distributions."
    ]

    templates = labeled_templates if problem["labeled_boxes"] else unlabeled_templates
    selected_template = random.choice(templates)
    
    # Add solution to the problem
    problem["distributions"] = distributions
    problem["distribution_count"] = count

    return Problem(
        question=selected_template.format(
            letters=problem["letters"], 
            num_boxes=len(problem["box_sizes"]), 
            box_sizes=problem["box_sizes"]
        ), 
        answer=count
    )


def debug_probability_problem(letter_counts, word_length, event_type, special_letter=None, n_specific_letters=None, shift=None):
    """
    Debug a specific probability problem with detailed output.
    
    Args:
        letter_counts: Dictionary of letter counts (e.g. {'i': 3, 'b': 3})
        word_length: Length of words to generate
        event_type: One of the event types in EVENT_CALCULATORS
        special_letter: Special letter for events requiring it
        n_specific_letters: Number of specific letters for events requiring it
        shift: Shift amount for events requiring it
        
    Returns:
        tuple: (words_list, success_count, total_events, probability fraction, m+n value)
    """
    # Generate all valid words with their probabilities
    words_dict = generate_all_words(word_length, letter_counts)
    words_list = list(words_dict.keys())
    
    # Prepare event parameters
    prob_kwargs = {
        'special_letter': special_letter,
        'n_specific_letters': n_specific_letters,
        'shift': shift
    }
    
    # Calculate probability
    probability = EVENT_CALCULATORS[event_type](words_dict, **prob_kwargs)
    
    # Get detailed counts and information
    success_count = 0
    total_events = 0
    
    if event_type in ['exactly_n_after_shift_m', 'at_least_n_after_shift_m']:
        # For shift-based events
        for word, (word_num, word_denom) in words_dict.items():
            shifted_word = shift_word(word, shift)
            
            if (event_type == 'exactly_n_after_shift_m' and 
                count_fixed_points_for_letter(word, shifted_word, special_letter) == n_specific_letters):
                success_count += word_num
            elif (event_type == 'at_least_n_after_shift_m' and 
                  count_fixed_points_for_letter(word, shifted_word, special_letter) >= n_specific_letters):
                success_count += word_num
                
            total_events += word_num
    else:
        # For permutation-based events
        for word, (word_num, word_denom) in words_dict.items():
            word_success = 0
            word_total = 0
            
            for perm in permutations(word):
                permuted_word = ''.join(perm)
                word_total += 1
                
                if event_type == 'no_fixed_points' and count_fixed_points(word, permuted_word) == 0:
                    word_success += 1
                elif event_type == 'no_specific_letter_fixed' and count_fixed_points_for_letter(word, permuted_word, special_letter) == 0:
                    word_success += 1
                elif event_type == 'exactly_n_specific_fixed' and count_fixed_points_for_letter(word, permuted_word, special_letter) == n_specific_letters:
                    word_success += 1
                elif event_type == 'at_least_n_specific_fixed' and count_fixed_points_for_letter(word, permuted_word, special_letter) >= n_specific_letters:
                    word_success += 1
            
            # Scale by the probability of sampling this word
            success_count += word_success * word_num
            total_events += word_total * word_denom
    
    # Format results
    num, denom = probability
    m_plus_n = num + denom
    
    # Print summary
    print(f"\n--- DEBUG PROBABILITY PROBLEM ---")
    print(f"Letter counts: {letter_counts}")
    print(f"Word length: {word_length}")
    print(f"Event type: {event_type}")
    if special_letter:
        print(f"Special letter: {special_letter}")
    if n_specific_letters is not None:
        print(f"N specific letters: {n_specific_letters}")
    if shift:
        print(f"Shift: {shift}")
    print(f"\nTotal words: {len(words_list)}")
    print(f"Words with probabilities:")
    for word, (num, denom) in words_dict.items():
        print(f"  {word}: {num}/{denom} = {num/denom:.4f}")
    print(f"\nSuccess count: {success_count}")
    print(f"Total events: {total_events}")
    print(f"Probability: {num}/{denom} = {num/denom if denom != 0 else 0}")
    print(f"m + n = {m_plus_n}")
    print("-" * 50)
    
    return words_list, success_count, total_events, probability, m_plus_n


# --- Regex Pattern Matching Functions ---

def count_pattern_matches(word, pattern, debug=False):
    """
    Count the number of matches of a regex pattern in a word.
    
    Args:
        word: The word to search in
        pattern: Compiled regex pattern object
        debug: If True, print debug information about matches
        
    Returns:
        int: Number of pattern matches
    """
    matches = []
    # For each possible start position
    for i in range(len(word)):
        # For each possible end position after the start
        for j in range(i+1, len(word)+1):
            substring = word[i:j]
            if pattern.fullmatch(substring):  # Use fullmatch to match the entire substring
                matches.append(substring)
    
    if debug:
        print(f"Word: {word}")
        print(f"Pattern: {pattern.pattern}")
        print(f"Matches: {matches}")
    
    return len(matches)

def generate_regex_pattern(available_letters=None):
    """
    Generate a random regex pattern for matching.
    
    Args:
        available_letters: List of letters to use in the pattern, if None random letters will be used
    
    Returns:
        tuple: (regex pattern object, pattern description string)
    """
    pattern_templates = [
        # Simple patterns
        (r"a.*b", "a followed by b"),
        (r"a.*a", "a followed by another a"),
        (r"b.*b", "b followed by another b"),
        (r"ab+", "a followed by one or more b's"),
        (r"a+b", "one or more a's followed by b"),
        (r"ab*", "a followed by zero or more b's"),
        (r"a*b", "zero or more a's followed by b"),
    ]
    
    # If we don't have enough letters, choose simpler patterns
    if available_letters and len(available_letters) < 2:
        # Filter to patterns that only need one letter type
        pattern_templates = [pt for pt in pattern_templates if 'b' not in pt[0]]
    
    # Select a pattern template
    pattern_str, description = random.choice(pattern_templates)
    
    # Determine how many unique letters we need
    unique_letters_needed = len(set(c for c in pattern_str if c in 'abc'))
    
    # Choose letters to use in the pattern
    if available_letters and len(available_letters) >= unique_letters_needed:
        # Use letters from the available set
        letters = random.sample(available_letters, unique_letters_needed)
    else:
        # Fall back to random letters if we don't have enough available letters
        letters = random.sample(string.ascii_lowercase, unique_letters_needed)
    
    # Create a mapping for replacement
    letter_map = {}
    letter_index = 0
    
    # Replace placeholders with actual letters
    new_pattern = pattern_str
    new_description = description
    
    for placeholder in 'abc':
        if placeholder in pattern_str:
            if placeholder not in letter_map:
                letter_map[placeholder] = letters[letter_index]
                letter_index += 1
            
            new_pattern = new_pattern.replace(placeholder, letter_map[placeholder])
            new_description = new_description.replace(placeholder, letter_map[placeholder])
    
    return re.compile(new_pattern), new_pattern, new_description

def calculate_expected_pattern_matches(letter_counts, length, pattern, debug=False):
    """
    Calculate the expected number of pattern matches in words formed by randomly selecting
    letters from the given multiset.
    
    Args:
        letter_counts: Dictionary of letter counts
        length: Word length
        pattern: Compiled regex pattern object
        debug: If True, print debug information
        
    Returns:
        float: Expected number of pattern matches
    """
    words_dict = generate_all_words(length, letter_counts)
    
    total_matches = 0
    total_probability = 0
    
    if debug:
        print(f"Calculating expected matches for pattern: {pattern.pattern}")
        print(f"Letter counts: {letter_counts}")
        print(f"Word length: {length}")
    
    for word, (word_num, word_denom) in words_dict.items():
        matches = count_pattern_matches(word, pattern, debug)
        word_prob = word_num / word_denom
        
        if debug:
            print(f"Word: {word}, Matches: {matches}, Probability: {word_prob}")
        
        total_matches += matches * word_prob
        total_probability += word_prob
    
    # Normalize by total probability (should be 1.0 if all words are considered)
    expected_matches = total_matches / total_probability if total_probability > 0 else 0
    
    if debug:
        print(f"Total matches (weighted): {total_matches}")
        print(f"Total probability: {total_probability}")
        print(f"Expected matches: {expected_matches}")
    
    return expected_matches

def generate_pattern_matching_problem(length=None, letters=None, letter_counts=None, 
                                      pattern=None, pattern_description=None, 
                                      random_seed=None, total_letters=None, debug=False):
    """
    Generate a problem about expected number of regex pattern matches.
    
    Args:
        length: Word length or None for random
        letters: List of letters to use or None for random
        letter_counts: Dict of letter counts or None for random
        pattern: Regex pattern or None for random
        pattern_description: Description of the pattern or None for auto-generated
        random_seed: Random seed for reproducibility or None
        total_letters: Total number of letters across all letter types
        debug: If True, print debug information
        
    Returns:
        tuple: (problem_statement, words, pattern_str, expected_matches, rounded_expected)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate or use provided length
    if length is None:
        length = random.randint(3, 5)
        
    # Set default total_letters if not provided
    if total_letters is None:
        total_letters = random.randint(8, 15)
        
    # Generate or use provided letters and counts
    if letters is None or letter_counts is None:
        available_letters = list(string.ascii_lowercase)
        if letters is None:
            num_letters = random.randint(2, 5)
            letters = random.sample(available_letters, num_letters)

        if letter_counts is None:
            # Generate counts that sum to the total
            counts = partition_integer(total_letters, len(letters), 1)
            letter_counts = dict(zip(letters, counts))
    
    # Generate or use provided pattern
    if pattern is None:
        # Pass the available letters to ensure the pattern uses letters from our set
        pattern, pattern_str, _ = generate_regex_pattern(list(letter_counts.keys()))
    else:
        pattern_str = pattern.pattern
    
    # Use the pattern string directly in the problem description
    pattern_description = f"the pattern '{pattern_str}'"
    
    # Calculate expected number of matches
    expected_matches = calculate_expected_pattern_matches(letter_counts, length, pattern, debug)
    
    # Generate a list of words for display
    words_dict = generate_all_words(length, letter_counts)
    words_list = list(words_dict.keys())
    
    # Create the problem statement
    letters_str = ', '.join([f"{letter}: {count}" for letter, count in letter_counts.items()])
    
    problem_templates = [
        "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}. What is the expected number of occurrences of {pattern_description} in each word? Round your answer to 2 decimal places.",
        "When randomly selecting {length} letters from the multiset {{{letters_str}}} to form a word, what is the expected number of matches of {pattern_description}? Round to 2 decimal places.",
        "If you form a word of length {length} by randomly selecting letters from the multiset {{{letters_str}}}, what is the expected number of times {pattern_description} appears? Round to 2 decimal places.",
    ]
    
    problem_statement = random.choice(problem_templates).format(
        length=length,
        letters_str=letters_str,
        pattern_description=pattern_description
    )
    
    # Round the expected matches to 2 decimal places
    rounded_expected_matches = round(expected_matches, 2)
    
    return problem_statement, words_list, pattern_str, expected_matches, rounded_expected_matches


# Main function for CLI usage
def main():
    """
    Command-line interface for generating combinatorics problems.
    """
    parser = argparse.ArgumentParser(description='Generate combinatorics problems.')
    parser.add_argument('--type', type=str, choices=['probability', 'distribution', 'pattern'], default='pattern',
                        help='Type of problem to generate (probability, distribution, or pattern)')
    parser.add_argument('--event', type=str, choices=['no_fixed_points', 'no_specific_letter_fixed', 
                                                      'exactly_n_specific_fixed', 'at_least_n_specific_fixed', 
                                                      'exactly_n_after_shift_m', 'at_least_n_after_shift_m', 'all'], 
                        default='all', help='Event type for probability problems')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--length', type=int, default=None, help='Word length for probability problems')
    parser.add_argument('--letters', type=str, default=None, help='Letters to use (comma-separated)')
    parser.add_argument('--total_letters', type=int, default=None, help='Total number of letters across all letter types')
    parser.add_argument('--labeled', action='store_true', help='Use labeled boxes for distribution problems')
    parser.add_argument('--num_boxes', type=int, default=None, help='Number of boxes for distribution problems')
    parser.add_argument('--num_letter_types', type=int, default=None, help='Number of letter types for distribution problems')
    parser.add_argument('--pattern', type=str, default=None, help='Regex pattern for pattern matching problems')
    parser.add_argument('--debug', action='store_true', help='Print debug information')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.type == 'probability':
        letters = None
        letter_counts = None
        if args.letters is not None:
            letters = args.letters.split(',')

        if args.event == 'all':
            # event_types = ['no_fixed_points', 'no_specific_letter_fixed', 'exactly_n_specific_fixed',
            #                 'at_least_n_specific_fixed', 'exactly_n_after_shift_m', 'at_least_n_after_shift_m']
            event_types = ['no_fixed_points', 'no_specific_letter_fixed', 'at_least_n_specific_fixed', 'at_least_n_after_shift_m']
        else:
            event_types = [args.event]

        for event_type in event_types:
            problem_statement, words, probability = generate_probability_problem(
                event_type=event_type,
                length=args.length,
                letters=letters,
                letter_counts=letter_counts,
                total_letters=args.total_letters
            )

            print(f"\n--- EVENT TYPE {event_type.upper()} ---")
            print(problem_statement)
            print(f"\nTotal number of valid words: {len(words)}")
            print(f"Words: {words}")
            
            # probability is now a tuple (numerator, denominator)
            num, denom = probability
            # Calculate decimal probability for display purposes
            decimal_prob = num / denom if denom != 0 else 0
            
            print(f"Probability: {decimal_prob:.6f}")
            print(f"Fraction representation: {num}/{denom}")
            print(f"m + n = {num + denom}")
            print("-" * 50)
    
    elif args.type == 'distribution':
        problem = distribute_letter_problem(
            num_letter_type_range=[args.num_letter_types] if args.num_letter_types else None,
            num_boxes_range=[args.num_boxes] if args.num_boxes else None,
            total_letters_range=[args.total_letters] if args.total_letters else None
        )
        print("\n--- LETTER DISTRIBUTION PROBLEM ---")
        print(f"Question: {problem.question}")
        print(f"Answer: {problem.answer}")
        print("-" * 50)
    
    elif args.type == 'pattern':
        letters = None
        letter_counts = None
        if args.letters is not None:
            letters = args.letters.split(',')

        pattern = None
        if args.pattern is not None:
            pattern = re.compile(args.pattern)
            
        problem_statement, words, pattern_str, expected_matches, rounded_expected = generate_pattern_matching_problem(
            length=args.length,
            letters=letters,
            letter_counts=letter_counts,
            pattern=pattern,
            total_letters=args.total_letters,
            debug=args.debug
        )

        print("\n--- PATTERN MATCHING PROBLEM ---")
        print(problem_statement)
        print(f"\nRegex pattern: {pattern_str}")
        print(f"Total number of valid words: {len(words)}")
        print(f"Sample words: {', '.join(words[:5])}..." if len(words) > 5 else f"Words: {', '.join(words)}")
        print(f"Expected number of matches: {expected_matches:.4f}")
        print(f"Rounded to 2 decimal places: {rounded_expected}")
        print("-" * 50)


if __name__ == "__main__":
    main()
    # debug_probability_problem({'i': 3, 'b': 3}, 3, "at_least_n_specific_fixed", special_letter="b", n_specific_letters=1)
    # debug_probability_problem({'a': 2, 'g': 5}, 3, "at_least_n_after_shift_m", special_letter="g", shift=2, n_specific_letters=2)


    