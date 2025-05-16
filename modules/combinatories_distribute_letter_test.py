from collections import Counter
from itertools import combinations


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


# Function to print distributions more clearly
def print_distributions(distributions, limit=None):
    print(f"Total distributions: {len(distributions)}")
    if limit and len(distributions) > limit:
        print(f"Showing first {limit} distributions:")
        for i, dist in enumerate(distributions[:limit]):
            print(f"{i + 1}. {dist}")
        print(f"... and {len(distributions) - limit} more distributions")
    else:
        for i, dist in enumerate(distributions):
            print(f"{i + 1}. {dist}")


# Example usage with the original problem
letters = {'c': 5, 'o': 3, 'a': 2}
box_sizes = [3, 3, 2, 2]

# Unlabeled boxes (default)
print("\nProblem with UNLABELED boxes: Distribute {c:5, o:3, a:2} into boxes of sizes [3, 3, 2, 2]")
unlabeled_distributions, unlabeled_count = find_letter_distributions(letters, box_sizes, labeled_boxes=False)
print(f"Number of ways with unlabeled boxes: {unlabeled_count}")
print_distributions(unlabeled_distributions, limit=1000)

# Labeled boxes
print("\nProblem with LABELED boxes: Distribute {c:5, o:3, a:2} into boxes of sizes [3, 3, 2, 2]")
labeled_distributions, labeled_count = find_letter_distributions(letters, box_sizes, labeled_boxes=True)
print(f"Number of ways with labeled boxes: {labeled_count}")
print_distributions(labeled_distributions, limit=1000)

# Test with a smaller example to show all distributions
small_test = {'a': 5, 'b': 3}
small_boxes = [3, 3, 2]

print("\n" + "=" * 80)
print(f"Smaller example: Distribute {small_test} into boxes of sizes {small_boxes}")

# Unlabeled boxes
small_unlabeled, small_unlabeled_count = find_letter_distributions(small_test, small_boxes, labeled_boxes=False)
print(f"Number of ways with unlabeled boxes: {small_unlabeled_count}")
print_distributions(small_unlabeled)

# Labeled boxes
small_labeled, small_labeled_count = find_letter_distributions(small_test, small_boxes, labeled_boxes=True)
print(f"Number of ways with labeled boxes: {small_labeled_count}")
print_distributions(small_labeled)