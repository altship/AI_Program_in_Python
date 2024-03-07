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


if __name__ == '__main__':
    a = [0, 1, 2, 3, 4, 5, 6]
    print(BinarySearch(a, -3))
