import random
import string
from collections import Counter
from itertools import combinations
import json
import example

# Set a random seed for reproducibility
RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)


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
    letters = rng.sample(string.ascii_lowercase, num_letter_types)

    if total_letters:
        # Generate counts that sum to the total
        counts = partition_integer(total_letters, num_letter_types, min_count)
    else:
        # Generate random counts
        counts = [rng.randint(min_count, max_count) for _ in range(num_letter_types)]

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
        box_sizes = [rng.randint(min_size, max_size) for _ in range(num_boxes)]

    return box_sizes


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
        idx = rng.randint(0, parts - 1)
        if result[idx] < max_val:
            result[idx] += 1
            remaining -= 1

    return result


def generate_problem(num_letter_types=None, num_boxes=None, total_letters=None, labeled_boxes=False):
    """
    Generate a complete letter distribution problem.

    Args:
        num_letter_types: Number of different letter types
        num_boxes: Number of boxes
        labeled_boxes: Whether boxes are labeled or not

    Returns:
        dict: Problem specification including letters, counts, and box sizes
    """
    if num_letter_types is None:
        num_letter_types = rng.randint(2, 5)

    if num_boxes is None:
        num_boxes = rng.randint(2, 6)

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
    Generate a problem and calculate all possible distributions.

    Args:
        num_letter_types: Number of different letter types
        num_boxes: Number of boxes
        labeled_boxes: Whether boxes are labeled or not

    Returns:
        dict: Problem specification and solution
    """

    labeled_boxes = True if random.random() < 0.5 else False
    num_letter_types = random.choice(num_letter_type_range)
    num_boxes = random.choice(num_boxes_range)
    total_letters = random.choice(total_letters_range)
    problem = generate_problem(num_letter_types, num_boxes, total_letters, labeled_boxes)

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

    return example.Problem(question=selected_template.format(letters=problem["letters"], num_boxes=len(problem["box_sizes"]), box_sizes=problem["box_sizes"]), answer=count)


# Example usage
if __name__ == "__main__":
    print("\nGenerating and solving a problem:")
    solved_problem = distribute_letter_problem(num_letter_type_range=(3, ), num_boxes_range=(3, ), total_letters_range=(18,))
    print((solved_problem))

    # Display a few distributions
    # max_to_show = min(5, len(solved_problem['distributions']))
    # print(f"\nFirst {max_to_show} distributions:")
    # for i, dist in enumerate(solved_problem['distributions'][:max_to_show]):
    #     print(f"Distribution {i + 1}: {dist}")

    # # Demonstrate original example
    # print("\nOriginal example:")
    # original = {
    #     "letters": {'c': 5, 'o': 3, 'a': 2},
    #     "box_sizes": [3, 3, 2, 2],
    #     "labeled_boxes": False,
    #     "problem_type": "unlabeled"
    # }
    # print(format_problem(original))
    #
    # # Solve the original example
    # distributions, count = find_letter_distributions(
    #     original["letters"],
    #     original["box_sizes"],
    #     original["labeled_boxes"]
    # )
    # print(f"Number of unique distributions: {count}")

    # Display a few distributions from original example
    # max_to_show = min(5, len(distributions))
    # print(f"\nFirst {max_to_show} distributions:")
    # for i, dist in enumerate(distributions[:max_to_show]):
    #     print(f"Distribution {i + 1}: {dist}")