from scipy.optimize import linprog
import numpy as np
import random
import time
import matplotlib.pyplot as plt


def set_cover(universe, subsets):
    selected_subsets = []
    remaining_elements = set(universe)
    while remaining_elements:
        best_subset = None
        best_subset_elements_covered = set()
        for subset in subsets:
            covered = remaining_elements & set(subset)
            if len(covered) > len(best_subset_elements_covered):
                best_subset = subset
                best_subset_elements_covered = covered
        if best_subset is None:
            return None
        selected_subsets.append(best_subset)
        remaining_elements -= best_subset_elements_covered
    return selected_subsets


def set_cover_lp(universe, subsets):
    num_universe = len(universe)
    num_subsets = len(subsets)
    c = np.ones(num_subsets)
    A = np.zeros((num_universe, num_subsets))
    b = np.ones(num_universe)
    for j, element in enumerate(universe):
        for i, subset in enumerate(subsets):
            if element in subset:
                A[j][i] = 1
    res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, 1.1), method="highs")
    cover = [subsets[i] for i in range(num_subsets) if res.x[i] > 1e-5]
    return cover, res


def generate_universe_and_subsets(size):
    universe = set(range(1, size + 1))
    subsets = []
    for _ in range(size):
        subset_size = random.randint(1, size // 10)
        subsets.append(set(random.sample(universe, subset_size)))
    return universe, subsets


def test_algorithm(algorithm, universe, subsets):
    start_time = time.time()
    result = algorithm(universe, subsets)
    end_time = time.time()
    return end_time - start_time, result


def perform_tests(sizes):
    times_set_cover = []
    times_set_cover_lp = []
    ans1 = []
    ans2 = []
    for size in sizes:
        universe, subsets = generate_universe_and_subsets(size)
        time_taken, res1 = test_algorithm(set_cover, universe, subsets)
        times_set_cover.append(time_taken)
        ans1.append(len(res1))
        time_taken, res2 = test_algorithm(set_cover_lp, universe, subsets)
        times_set_cover_lp.append(time_taken)
        ans2.append(len(res2))
    return times_set_cover, times_set_cover_lp, ans1, ans2


def plot_performance(sizes, times_set_cover, times_set_cover_lp):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times_set_cover, label="set_cover", marker="o")
    plt.plot(sizes, times_set_cover_lp, label="set_cover_lp", marker="x")
    plt.xlabel("Size of Universe")
    plt.ylabel("Time taken (seconds)")
    plt.title("Performance of Set Cover Algorithms")
    plt.legend()
    plt.grid(True)
    plt.savefig("set_cover_performance.png")
    plt.show()


def plot_ans(sizes, ans1, ans2):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ans1, label="set_cover", marker="o")
    plt.plot(sizes, ans2, label="set_cover_lp", marke="x")
    plt.xlabel("Size of Universe")
    plt.ylabel("anwser len")
    plt.title("Performance of Set Cover Algorithms")
    plt.legend()
    plt.grid(True)
    plt.savefig("set_cover_performance2.png")
    plt.show()


sizes = [100, 1000, 5000]
times_set_cover, times_set_cover_lp, asn1, ans2 = perform_tests(sizes)
plot_performance(sizes, times_set_cover, times_set_cover_lp)
plot_ans()
