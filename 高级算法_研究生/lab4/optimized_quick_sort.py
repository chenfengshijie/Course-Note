import random
import heapq
from math import log


def quick_sort(A, p, r, max_depth):
    size = r - p + 1

    # 使用插入排序当区间小于等于5
    if size <= 5:
        insertion_sort(A, p, r)
        return

    # 如果递归深度超过限制，使用堆排序
    if max_depth == 0:
        heap_sort(A, p, r)
        return

    if p < r:
        lt, gt = three_way_partition(A, p, r)
        # 递归处理左右子数组，减小最大深度
        quick_sort(A, p, lt - 1, max_depth - 1)
        quick_sort(A, gt + 1, r, max_depth - 1)


def insertion_sort(A, p, r):
    for i in range(p + 1, r + 1):
        key = A[i]
        j = i - 1
        while j >= p and A[j] > key:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key


def heap_sort(A, p, r):
    heapq.heapify(A[p : r + 1])
    sorted_array = [heapq.heappop(A[p : r + 1]) for _ in range(p, r + 1)]
    A[p : r + 1] = sorted_array


def three_way_partition(A, p, r):
    pivot = A[r]
    lt = p
    gt = r
    i = p
    while i <= gt:
        if A[i] < pivot:
            A[lt], A[i] = A[i], A[lt]
            lt += 1
            i += 1
        elif A[i] > pivot:
            A[gt], A[i] = A[i], A[gt]
            gt -= 1
        else:
            i += 1
    return lt, gt


# 获取数组的最大递归深度
def get_max_depth(length):
    return int(log(length) * 2)


if __name__ == "__main__":

    arr = [1, 8, 3, 5, 6, 3, 2, 5, 6, 6, 5, 5, 3, 7, 8, 2, 3, 1, 1, 1]
    max_depth = get_max_depth(len(arr))
    quick_sort(arr, 0, len(arr) - 1, max_depth)
    print(arr)
