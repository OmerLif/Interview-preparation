import random
import re
from typing import List, Tuple, Dict, Deque
from math import inf
from collections import Counter, defaultdict
from functools import reduce
from itertools import repeat
import heapq

"""
    this is a solution for many of the problems in cracking the code interview and leetcode 
    Top interview Questions.
    I will be using the python 3.6.4 version of python.
    I will be using the following libraries:
    - collections
    - re
    - typing
    - copy
    - itertools
    - math
    - heapq
"""

############### Some implementation of the sorting algorithms ###############


##### Quick sort implementation #####
def partition(arr, l, r) -> int:
    # get a random integer in the range of l to r
    pivot = random.randint(l, r)
    # swap the pivot with the last element
    arr[pivot], arr[r] = arr[r], arr[pivot]
    # set the pivot to the first element the last element that is less than the pivot
    pivot = l
    # iterate through the array
    for i in range(l, r):
        # if the element is less than the pivot
        if arr[i] < arr[r]:
            # swap the element with the pivot
            arr[i], arr[pivot] = arr[pivot], arr[i]
            # increment the pivot
            pivot += 1
    # swap the pivot with the last element
    arr[pivot], arr[r] = arr[r], arr[pivot]
    # return the pivot
    return pivot


# Runs in O(n log n) time and O(n) space
def quick_sort(arr, l, r):
    if (r <= l or not arr):
        return
    m = partition(arr, l, r)
    quick_sort(arr, l, m - 1)
    quick_sort(arr, m + 1, r)


# Implementing binary search classic algorithm. Runs in O(log n) time and O(1) space.
def binary_search(arr, elem, l, r):
    if l >= r:
        return False
    if elem == arr[int((l + r) / 2)]:
        return int(l + r) / 2
    elif elem < arr[int((l + r) / 2)]:
        return binary_search(arr, elem, l, int((l + r) / 2))
    else:
        return binary_search(arr, elem, int((l + r) / 2) + 1, r)


##### Merge sort implementation #####
# The merge function takes two sorted arrays and merges them into one sorted array. Runs in O(n) time and O(n) space.
def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


# Runs in O(n log n) time and O(n) space.
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


##### Selectio sort implementation #####
# Runs in O(n**2) time and O(1) space.
def selection_sort(lst):
    for i in range(len(lst) - 1):
        min_ind = i
        for j in range(i, len(lst)):
            if lst[j] < lst[min_ind]:
                min_ind = j
        lst[i], lst[min_ind] = lst[min_ind], lst[i]
    return lst


##### Bubble sort implementation #####
# Sorting in O(n**2) time and O(1) space.
def bubble_sort(lst):
    for i in range(len(lst)):
        swap = False
        for j in range(len(lst) - i - 1):
            if lst[j + 1] < lst[j]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swap = True
        if not swap:
            break
    return lst


##### Counting sort implementation #####
# Sorting in O(n) time and O(n) space. base on the assumption that the values in the array are between 0 and k.
def counting_sort(arr: List[int], k: int) -> List[int]:
    counting_arr = [0] * (k + 1)
    result_arr = [None] * len(arr)
    for num in arr:
        counting_arr[num] += 1
    for i in range(1, k + 1):
        counting_arr[i] += counting_arr[i - 1]

    print("counting_arr:", counting_arr)
    for j in range(len(arr) - 1, -1, -1):
        result_arr[counting_arr[arr[j]] - 1] = arr[j]
        counting_arr[arr[j]] -= 1
    return result_arr


##### Radix sort implementation #####
# Sorting in O(n) time and O(n) space. base on the assumption that the amount of digits in the
# values in the array is between 0 and k.
def radix_sort(arr: List[int]) -> List[int]:
    max_num = max(arr)
    max_digit = len(str(max_num))
    for i in range(max_digit):
        bucket = [[] for _ in range(10)]
        for num in arr:
            digit = (num // (10 ** i)) % 10
            bucket[digit].append(num)
        arr = []
        for bucket_arr in bucket:
            arr.extend(bucket_arr)
        print("bucket:", arr)
    return arr


#####################################################################################################################
# You are a product manager and currently leading a team to develop a new product.
# Unfortunately, the latest version of your product fails the quality check.
# Since each version is developed based on the previous version, all the versions after a bad version are also bad.
# Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one,
# which causes all the following ones to be bad.
# You are given an API bool isBadVersion(version) which returns whether version is bad.
# Implement a function to find the first bad version. You should minimize the number of calls to the API.

# The isBadVersion API is already defined for you.
def isBadVersion(version: int) -> bool:
    pass


def firstBadVersion(n: int) -> int:
    first_bad = 1

    # while we are not crossing the n boundry and the verstion is ok
    while first_bad < n and not isBadVersion(first_bad):
        first_bad *= 2
    low = first_bad // 2
    high = first_bad
    while low <= high:
        mid = (low + high) // 2
        if isBadVersion(mid):
            high = mid - 1
        elif not isBadVersion(mid):
            low = mid + 1
    return low


###### Questions from leetcode.com Top 50 most common interview Questions ######

#####################################################################################################################
# Given an array nums with n objects colored red, white, or blue, sort them in-place
# so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
# We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
# You must solve this problem without using the library's sort function.
def sortColors(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    # The idea of this solution come from the partition algo
    l = -1
    r = len(nums)  # blue pointer
    i = 0
    while i < r:
        if nums[i] == 0:
            l += 1
            nums[i], nums[l] = nums[l], nums[i]
            i += 1
        elif nums[i] == 1:
            i += 1
        else:
            # nums[i] == 2
            r -= 1
            nums[i], nums[r] = nums[r], nums[i]


#####################################################################################################################
# Given an integer array nums and an integer k, return the k most frequent elements.
# There are number of possible answers in the comments.
def topKFrequent(nums: List[int], k: int) -> List[int]:
    return [key for key, val in Counter(nums).most_common(k)]

    # def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    # counter_dict = Counter(nums)
    # n = len(nums)
    # freq_lst = [[] for i in repeat(None, n+1)]
    # res = []

    # def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    # for key, value in counter_dict.items():
    # freq_lst[value].append(key)
    # index = -1
    # while k > 0:
    #   res.extend(freq_lst[index])
    #   k -= len(freq_lst[index])
    #   index -= 1
    #   return res


#####################################################################################################################
# Given an integer array nums and an integer k, return the kth largest element in the array.
# Note that it is the kth largest element in the sorted order, not the kth distinct element.
def findKthLargest(nums: List[int], k: int) -> int:
    return heapq.nlargest(k, nums)[-1]
    # nums = list(-np.array(nums))
    # heapq.heapify(nums)
    # res = 0
    # while k > 0:
    #     res = heapq.heappop(nums)
    #     print(res)
    #     k -= 1
    # return -res


#####################################################################################################################
# A peak element is an element that is strictly greater than its neighbors.
# Given a 0-indexed integer array nums, find a peak element, and return its index.
# If the array contains multiple peaks, return the index to any of the peaks.
# You may imagine that nums[-1] = nums[n] = -∞. In other words,
# an element is always considered to be strictly greater than a neighbor that is outside the array.
def findPeakElement(nums: List[int]) -> int:
    def helper(nums, l, r):
        m = (l + r) // 2
        if is_peak(nums, m):
            return m
        if m > 0 and nums[m] < nums[m - 1]:
            return helper(nums, l, m - 1)
        if m < (len(nums) - 1) and nums[m] < nums[m + 1]:
            return helper(nums, m + 1, r)

    def is_peak(nums, ind):
        if (ind == 0 and nums[ind] > nums[ind + 1]) \
                or (ind == len(nums) - 1 and nums[ind] > nums[ind - 1]) \
                or (0 < ind < (len(nums) - 1) and nums[ind] > nums[ind - 1] and nums[ind] > nums[ind + 1]):
            return True
        return False

    if len(nums) == 1:
        return 0
    return helper(nums, 0, len(nums) - 1)


# Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.
# If target is not found in the array, return [-1, -1].
def searchRange(nums: List[int], target: int) -> List[int]:
    first = last = perv = -1
    res = []

    if not nums:
        return first, perv

    def binary_search(arr, low, high, x):
        # Check base case
        if high >= low:

            mid = (high + low) // 2

            # If element is present at the middle itself
            if arr[mid] == x:
                return mid

            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif arr[mid] > x:
                return binary_search(arr, low, mid - 1, x)

            # Else the element can only be present in right subarray
            else:
                return binary_search(arr, mid + 1, high, x)

        else:
            # Element is not present in the array
            return -1

    n = len(nums) - 1
    mid = binary_search(nums, 0, n, target)
    if nums[0] != target:
        first = mid
        while first != -1:
            perv = first
            first = binary_search(nums, 0, first - 1, target)
        first = perv
    else:
        first = 0
    if nums[-1] != target:
        last = mid
        while last != -1:
            perv = last
            last = binary_search(nums, last + 1, n, target)
        last = perv
    else:
        last = n

    return first, last


#####################################################################################################################
# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals,
# and return an array of the non-overlapping intervals that cover all the intervals in the input.
def merge_intervals(self, intervals: List[List[int]]) -> List[List[int]]:
    # Sorting the intervals base on the start index
    intervals.sort(key=lambda i: i[0])
    result = [intervals[0]]

    for start, end in intervals[1:]:
        # the end "time" from the last elem in result
        last_end = result[-1][1]

        # We have a overlap
        if last_end >= start:
            # updating the end point of the last interval
            result[-1][1] = max(last_end, end)
        else:
            # no overlap
            result.append([start, end])
    return result


#####################################################################################################################
# There is an integer array nums sorted in ascending order (with distinct values).
# Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length)
# such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed).
# For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
# Given the array nums after the possible rotation and an integer target, return the index of target
# if it is in nums, or -1 if it is not in nums.
def search_in_rotated_sorted_array(nums: List[int], target: int) -> int:
    l = 0
    r = len(nums) - 1

    while l <= r:
        m = (l + r) // 2

        if nums[m] == target:
            return m
        # m is part of the left sorted sub array
        if nums[m] >= nums[l]:
            if nums[m] >= target and nums[l] <= target:
                r = m - 1
            elif nums[m] >= target and nums[l] >= target:
                l = m + 1
            elif nums[m] <= target:
                l = m + 1
        else:
            if nums[m] >= target:
                r = m - 1
            elif nums[m] <= target and target <= nums[r]:
                l = m + 1
            elif nums[m] <= target and target >= nums[r]:
                r = m - 1
    return -1


#####################################################################################################################
# Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix.
# This matrix has the following properties:
# Integers in each row are sorted in ascending from left to right.
# Integers in each column are sorted in ascending from top to bottom.
def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    top = 0
    cols = len(matrix[0])
    rows = len(matrix)
    right = cols - 1
    while top < rows and right >= 0:
        if matrix[top][right] == target:
            return True
        elif matrix[top][right] < target:
            top += 1
        else:
            right -= 1
    return False


def test_bubble_sort():
    assert bubble_sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert bubble_sort([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert bubble_sort([]) == []
    assert bubble_sort([1]) == [1]
    assert bubble_sort([1, 1]) == [1, 1]
    assert bubble_sort([2, 1]) == [1, 2]


def test_merge_sort():
    assert merge_sort([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert merge_sort([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert merge_sort([]) == []
    assert merge_sort([1]) == [1]
    assert merge_sort([1, 1]) == [1, 1]
    assert merge_sort([2, 1]) == [1, 2]


def test_counting_sort():
    print("counting_sort:", counting_sort([0, 1, 2, 1, 0, 2, 1, 0], 3))
    print("counting_sort:", counting_sort([0, 0, 0, 1, 2, 1, 0, 2, 1, 0], 3))
    print("counting_sort:", counting_sort([0, 0, 0, 1, 2, 1, 0, 2, 1, 0, 0, 0, 0], 3))
    print("counting_sort:", counting_sort([2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 3))
    print("counting_sort:", counting_sort([2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 0, 0, 1], 5))


def test_radix_sort():
    print("radix_sort:", radix_sort([0, 1, 2, 1, 0, 2, 1, 0]))
    print("radix_sort:", radix_sort([0, 0, 0, 1, 2, 1, 0, 2, 1, 0, 0, 0, 0]))
    print("radix_sort:", radix_sort([2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
    print("radix_sort:", radix_sort([2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 0, 0, 1]))
    print("radix_sort:", radix_sort([1, 4, 2, 3, 12, 23, 345, 567, 34, 22, 1, 24, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
