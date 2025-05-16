from itertools import product, permutations
from collections import Counter
import random
import argparse
import math

random.seed(0)

# Utility functions
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


# Event templates and calculators
EVENT_DESCRIPTIONS = {
    'no_fixed_points': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of no same letter remains in its original position?",
    'no_specific_letter_fixed': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of no letter '{special_letter}' remains in its original position?",
    'exactly_n_specific_fixed': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of exact {n_specific_letters} letter '{special_letter}' remains in its original position?",
    'at_least_n_specific_fixed': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shuffle the letters in the word, what is the probability of at least {n_specific_letters} letter '{special_letter}' remains in its original position?",
    'exactly_n_after_shift_m': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shift the letter by {shift} position {example}, what is the probability of exact {n_specific_letters} letter '{special_letter}' remains in its original position?",
    'at_least_n_after_shift_m': "Form a word by randomly choosing {length} letters from the multiset {{{letters_str}}}, shift the letter by {shift} position {example}, what is the probability of at least {n_specific_letters} letter '{special_letter}' remains in its original position?",
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


# Event-specific functions for multiset-based sampling
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


def generate_problem(event_type=None, length=None, letters=None, letter_counts=None,
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
    if letters is None or letter_counts is None:
        available_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        if letters is None:
            num_letters = random.randint(2, 4)
            letters = random.sample(available_letters, num_letters)

        if letter_counts is None:
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


def main():
    """
    Command-line interface for generating probability problems.
    """
    parser = argparse.ArgumentParser(description='Generate probability problems about word permutations.')
    parser.add_argument('--event', type=str, choices=['no_fixed_points', 'no_specific_letter_fixed', 
                                                      'exactly_n_specific_fixed', 'at_least_n_specific_fixed', 
                                                      'exactly_n_after_shift_m', 'at_least_n_after_shift_m', 'all'], default='all',
                        help='Event type to generate (no_fixed_points, no_specific_letter_fixed, exactly_n_specific_fixed, at_least_n_specific_fixed, exactly_n_after_shift_m, at_least_n_after_shift_m, or all)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--length', type=int, default=None, help='Word length')
    parser.add_argument('--letters', type=str, default=None, help='Letters to use (comma-separated)')
    parser.add_argument('--total_letters', type=int, default=None, help='Total number of letters across all letter types')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

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
        problem_statement, words, probability = generate_problem(
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


if __name__ == "__main__":
    main()