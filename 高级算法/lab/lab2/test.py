import numpy as np


def quicksort(arr, left, right):
    if left >= right:
        return
    pivot = arr[left]
    i = left
    j = right
    while i < j:
        while i < j and arr[j] >= pivot:
            j -= 1
        arr[i] = arr[j]
        while i < j and arr[i] <= pivot:
            i += 1
        arr[j] = arr[i]
    arr[i] = pivot
    quicksort(arr, left, i - 1)
    quicksort(arr, i + 1, right)