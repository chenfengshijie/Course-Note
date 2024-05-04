from optimized_quick_sort import quick_sort as optimized_sort
from origin_quick_sort import quick_sort as origin_sort
import time
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from math import log2


# 测量排序算法的运行时间
def measure_time(sort_func: Callable, A: np.array, optimizerd=False):
    start_time = time.time()
    try:
        if optimizerd:
            sort_func(A, 0, len(A) - 1, log2(len(A)))
        else:
            sort_func(A, 0, len(A) - 1)
    except Exception:
        return 100
    return time.time() - start_time


dataset_sizes = 10**6
repetitions = [0.1 * i for i in range(11)]
original_times = []
optimized_times = []


for rep in repetitions:
    # 生成具有指定重复百分比的数据集
    num_unique = int(dataset_sizes * (1 - rep))
    num_repeated = dataset_sizes - num_unique
    unique_elements = np.arange(num_unique + 10)
    repeated_elements = np.random.choice(unique_elements, size=num_repeated)
    dataset = np.concatenate((unique_elements, repeated_elements))
    np.random.shuffle(dataset)
    # print(len(dataset))

    dataset_copy = dataset.copy()
    original_time = measure_time(origin_sort, dataset_copy)
    # print(dataset_copy)
    original_times.append(original_time)

    dataset_copy = dataset.copy()
    optimized_time = measure_time(optimized_sort, dataset_copy, True)
    # print(dataset_copy)
    optimized_times.append(optimized_time)

plt.figure(figsize=(10, 6))
plt.plot(repetitions, original_times, label="Original QuickSort")
plt.plot(repetitions, optimized_times, label="Optimized QuickSort")
plt.xlabel("Percentage of Repetitions")
plt.ylabel("Time (seconds)")
plt.title("QuickSort Performance Comparison")
plt.legend()
plt.grid(True)
plt.savefig("quicksort_comparison.png")
plt.show()
