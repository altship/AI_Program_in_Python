# Lab Week 01 Assignment, From CSE SYSU

from typing import *


def BinarySearch(nums: List[int], target: int) -> int:
    i = 0
    j = len(nums) - 1
    while i < j:
        mid = (i + j) // 2
        if nums[mid] < target:
            i = mid + 1
        else:
            j = mid
    return i if nums[i] == target else -1


def MatrixAdd(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    m = len(A[0])
    ans = [[0] * m for _ in range(n)]  # Create a new list to store result to avoid change origin data.
    for i in range(n):
        for j in range(m):
            ans[i][j] = A[i][j] + B[i][j]
    return ans


def MatrixMul(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    m = len(A[0])
    k = len(B[0])
    ans = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            ans[i][j] = sum(A[i][x] * B[x][j] for x in range(m))
    return ans


def ReverseKeyValue(dict1: dict) -> dict:
    redict = {}
    for key, value in dict1.items():
        redict[value] = key
    return redict


if __name__ == '__main__':
    a = [0, 1, 2, 3, 4, 5, 6]
    print(BinarySearch(a, -3))
    A = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    B = [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]
    C = MatrixAdd(A, B)
    D = MatrixMul(A, B)
    dict1 = {'Alice': '001', 'Bob': '002', 'Calvin': '003'}
    dict2 = ReverseKeyValue(dict1)
    print(f"{len(A)}, {len(A[0])}")
    for i in C:
        for j in i:
            print(f"{j} ", end="")
        print()
    print()
    for i in D:
        for j in i:
            print(f"{j} ", end="")
        print()
    print(dict2)
