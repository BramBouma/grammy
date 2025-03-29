from nltk.corpus import words
import collections
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Download necessary corpora (only needed once)
# nltk.download('words')
# nltk.download('brown')


def can_form(word, available_letters):
    """Check if the word can be formed with the available letters."""
    word_count = collections.Counter(word)
    letters_count = collections.Counter(available_letters)
    return all(letters_count[letter] >= count for letter, count in word_count.items())


def recursive_search(remaining, start, current, filtered_words):
    """
    Recursively search for solutions that use up the remaining letters.
    Returns a list of solutions (each solution is a tuple of words).
    """
    solutions = []
    if sum(remaining.values()) == 0:
        solutions.append(tuple(current))
        return solutions
    for i in range(start, len(filtered_words)):
        word = filtered_words[i]
        word_counter = collections.Counter(word)
        if all(
            remaining.get(letter, 0) >= count for letter, count in word_counter.items()
        ):
            new_remaining = remaining.copy()
            new_remaining.subtract(word_counter)
            # Clean up zero and negative counts.
            new_remaining = +new_remaining
            current.append(word)
            solutions.extend(
                recursive_search(new_remaining, i, current, filtered_words)
            )
            current.pop()
    return solutions


def worker(args):
    """
    Worker function for multiprocessing.
    Each worker takes a candidate index from the filtered_words list,
    subtracts its letters from the available pool, and then continues the recursive search.
    """
    i, available, filtered_words = args
    word = filtered_words[i]
    word_counter = collections.Counter(word)
    # If the candidate word isn't valid, return an empty list.
    if not all(
        available.get(letter, 0) >= count for letter, count in word_counter.items()
    ):
        return []
    new_remaining = available.copy()
    new_remaining.subtract(word_counter)
    new_remaining = +new_remaining
    return recursive_search(new_remaining, i, [word], filtered_words)


def find_anagram_solutions(letters, freq_threshold=5):
    """
    Returns a list of solutions, where each solution is a tuple of words
    that together use all the letters exactly once.
    Only words of at least 4 characters that are common (appear at least freq_threshold times
    in the Brown corpus) are considered.
    The search is parallelized using multiprocessing.
    """
    # Normalize letters and count available letters.
    letters = [letter.lower() for letter in letters]
    available = collections.Counter(letters)

    # Build a set of common words from the Brown corpus.
    # Here, we only consider alphabetic tokens and lowercase them.
    # brown_words = [w.lower() for w in brown.words() if w.isalpha()]
    # brown_freq = nltk.FreqDist(brown_words)
    # common_set = {word for word, freq in brown_freq.items() if freq >= freq_threshold}

    # Filter the NLTK words corpus: word must be at least 4 letters, can be formed from the available letters,
    # and must be in the common_set.
    word_list = words.words()
    # filtered_words = [
    #     word.lower() for word in word_list
    #     if len(word) >= 4 and can_form(word.lower(), available) and word.lower() in common_set
    # ]
    filtered_words = [
        word.lower()
        for word in word_list
        if len(word) >= 4 and can_form(word.lower(), available)
    ]
    filtered_words = sorted(set(filtered_words))

    results = []
    # Prepare arguments for each candidate word at the first level.
    pool_args = [(i, available, filtered_words) for i in range(len(filtered_words))]

    # Use multiprocessing Pool to distribute the search across CPU cores.
    with Pool(cpu_count()) as pool:
        # Use imap_unordered to iterate over results as they complete.
        for sol in tqdm(
            pool.imap_unordered(worker, pool_args),
            total=len(pool_args),
            desc="Multiprocessing anagram search",
        ):
            results.extend(sol)
    return results


if __name__ == "__main__":
    letters = input("Enter letters separated by spaces: ").split()
    solutions = find_anagram_solutions(letters)

    print(f"\nAnagram solutions ({len(solutions)} found):")

    for sol in solutions:
        print(sol)

    print(f"\nAnagram solutions ({len(solutions)} found)")
