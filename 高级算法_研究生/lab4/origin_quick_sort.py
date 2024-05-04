import random
import time
import numpy as np


def quick_sort(A: np.array, p: int, r: int):

    if p < r:
        q = partition(A, p, r)
        quick_sort(A, p, q - 1)
        quick_sort(A, q + 1, r)


def partition(A, p, r):
    i = A[p]
    A[i], A[r] = A[r], A[i]
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


if __name__ == "__main__":
    A = np.random.randint(100, size=30)
    start_time = time.time()
    quick_sort(A, 0, len(A) - 1)
    print(A)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
