from nltk.corpus import words, brown
import collections
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Download necessary corpora (only needed once)
# nltk.download('words')
# nltk.download('brown')


def can_form(word, available_letters):
    """check if the word can be formed with the available letters."""
    word_count = collections.Counter(word)
    letters_count = collections.Counter(available_letters)
    return all(letters_count[letter] >= count for letter, count in word_count.items())


def recursive_search(remaining, start, current, filtered_words):
    """
    recursively search for solutions that use up the remaining letters.
    returns a list of solutions (each solution is a tuple of words).
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

            new_remaining = +new_remaining
            current.append(word)
            solutions.extend(
                recursive_search(new_remaining, i, current, filtered_words)
            )
            current.pop()
    return solutions


def worker(args):
    """
    worker function for multiprocessing.
    each worker takes a candidate index from the filtered_words list,
    subtracts its letters from the available pool, and then continues the recursive search.
    """
    i, available, filtered_words = args
    word = filtered_words[i]
    word_counter = collections.Counter(word)
    # if the candidate word isn't valid, return an empty list.
    if not all(
        available.get(letter, 0) >= count for letter, count in word_counter.items()
    ):
        return []

    new_remaining = available.copy()
    new_remaining.subtract(word_counter)
    new_remaining = +new_remaining

    return recursive_search(new_remaining, i, [word], filtered_words)


def find_anagram_solutions(letters, use_common=False, freq_threshold=4):
    """
    returns a list of solutions, where each solution is a tuple of words
    that together use all the letters exactly once
    only words with at least 4 characters are considered
    option to filter for words that are somewhat common (appear at least freq_threshold times
    in the brown corpus)
    """
    letters = [letter.lower() for letter in letters]
    available = collections.Counter(letters)

    # build a set of common words from the brown corpus if requested.
    if use_common:
        brown_words = [w.lower() for w in brown.words() if w.isalpha()]
        brown_freq = collections.Counter(brown_words)
        common_set = {
            word for word, freq in brown_freq.items() if freq >= freq_threshold
        }

    # filter the nltk words corpus: word must be at least 4 letters and formable from available letters.
    filtered_words = []
    word_list = words.words()
    for word in word_list:
        w = word.lower()
        if len(w) < 4:
            continue
        if not can_form(w, available):
            continue
        if use_common and w not in common_set:
            continue
        filtered_words.append(w)
    filtered_words = sorted(set(filtered_words))

    results = []
    pool_args = [(i, available, filtered_words) for i in range(len(filtered_words))]

    # use multiprocessing pool to distribute the search across cpu cores.
    with Pool(cpu_count()) as pool:
        # use imap_unordered to iterate over results as they complete.
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
