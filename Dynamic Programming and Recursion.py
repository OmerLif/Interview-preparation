import random
import re
from typing import List, Tuple, Dict, Deque
from math import inf
from collections import Counter, defaultdict
from functools import reduce
import numpy as np

"""
    this is a solution for many of the problems in cracking the code interview and leetcode 
    Top interview Questions.
    I will be using the python 3.6.4 version of python.
    I will be using the following libraries:
    - collections
    - re
    - typing
    - copy
"""


###### Recursion and Dynamic Programming #######

# the amount of ways to climb a staircase of n steps
# f(n) = f(n-1) + f(n-2) + f(n-3)
# that's like a "fibonacci" sequence so we can use memoization.
# Time Complexity: O(n), Space Complexity: O(1)
def triple_stairs(n):
    if n < 0:
        return 0
    base_0 = 1
    base_1 = 1
    base_2 = 2
    res = 0
    for i in range(2, n):
        res = base_0 + base_1 + base_2
        base_0 = base_1
        base_1 = base_2
        base_2 = res
    return res


#####################################################################################################################
GRID = List[List[int]]
NO_PASS = -1


# This function returns all the paths from the top left corner to the bottom right corner of a grid.
# The grid is represented as a list of lists.
def robot_in_grid_all_paths(grid: GRID):
    result = []
    visited_points = defaultdict(lambda: False)

    # Edge cases
    if not grid or grid[-1][-1] == NO_PASS:
        return []
    if len(grid) == 1:
        if any(elem == NO_PASS for elem in grid[0]):
            return []

    # inner dfs recursive function
    def _helper(down_counter: int, right_counter: int) -> None:

        # base cases where we succeed or didnt succeed in our quest
        if down_counter == len(grid) - 1 and right_counter == len(grid[0]) - 1:
            print("Found a path: ", result)
            return result
        if down_counter > len(grid) - 1 or right_counter > len(grid[0]) - 1:
            print("There is no path!")
            return

        curr_point = grid[down_counter][right_counter]
        if visited_points[curr_point]:
            return
        # Moving down
        if down_counter < len(grid) - 1 and grid[down_counter + 1][right_counter] != NO_PASS:
            result.append('D')
            _helper(down_counter + 1, right_counter)
            result.pop()

        # Moving right
        if right_counter < len(grid[0]) - 1 and grid[down_counter][right_counter + 1] != NO_PASS:
            result.append('R')
            _helper(down_counter, right_counter + 1)
            result.pop()

        return result

    return _helper(0, 0)


# This function checks whether there is a path from the top left corner to the bottom right corner of a grid.
# This solution
def robot_in_grid(grid: GRID) -> List:
    path = []

    # Used for memorization like in dfs
    visited_points = defaultdict(lambda: False)

    # A helper function that search if there is a path from the current point to the bottom right corner of the grid.
    def has_path(row: int, col: int) -> bool:
        if row < 0 or col < 0 or grid[row][col] == NO_PASS:
            return False
        curr_elem = grid[row][col]
        if visited_points[curr_elem]:
            return False

        # Edge case is case we are at the origin
        in_origin = row == 0 and col == 0
        if has_path(row - 1, col) or has_path(row, col - 1) or in_origin:
            path.append((row, col))
            return True

        visited_points[curr_elem] = True
        return False

    if has_path(len(grid) - 1, len(grid[0]) - 1):
        print("Found a path for the robot: ", path)


#####################################################################################################################
# This function gets a sorted list of numbers and returns whether there is number that equals to its index.
# The solution is to use a binary search.
# Time Complexity: O(log n), Space Complexity: O(1).
def magic_index(nums: List[int]) -> int:
    def _helper(nums, l, r):
        if l <= r:
            mid = (r + l) // 2
            if nums[mid] == mid:
                return mid
            if nums[mid] < mid:
                return _helper(nums, mid + 1, r)
            if nums[mid] > mid:
                return _helper(nums, l, mid - 1)
        return -1

    print(_helper(nums, 0, len(nums) - 1))


#####################################################################################################################
# This function creates all the possible sub set of a set. The solution is using backtracking.
# Time Complexity: O(2^n), Space Complexity: O(2^n).
def power_set(set):
    res = []
    subset = []

    # Create a recursive function that get the index of element in the set
    # and in each call one time it will add the element to the subset and
    # at the second call it will remove the element from the subset
    def all_subs(index):
        if index >= len(set):  # we made len(set) decisions
            res.append(subset[:])
            return

        # include the current element increment the index to indicate that we made a decision
        subset.append(set[index])
        all_subs(index + 1)

        # exclude the current element increment the index to indicate that we made a decision
        subset.pop()
        all_subs(index + 1)

    all_subs(0)
    return res


#####################################################################################################################
# This function find all the permutations of a string. The solution is using backtracking.
# Time Complexity: O(n!), Space Complexity: O(n!).
def string_permutations(string: str):
    if len(string) == 1:
        return [string]
    else:
        res = []
        for i in range(len(string)):
            for perm in string_permutations(string[:i] + string[i + 1:]):
                res.append(string[i] + perm)
        return res


#####################################################################################################################
class TreeNode():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


# This function checks if Two Trees are a rotation of each other.
def rotated_trees(root_1: TreeNode, root_2) -> bool:
    if not root_1 and not root_2:
        return True
    if not root_1 or not root_2:
        return False
    return root_1.val == root_2.val and rotated_trees(root_1.left, root_2.right) and rotated_trees(root_1.right,
                                                                                                   root_2.left)


#####################################################################################################################
# longest common subsequence using dynamic programming.
def max_sub_String_1(str_1, str_2):
    tabel = [[0 for i in range(len(str_2) + 1)] for j in range(len(str_1) + 1)]
    for i in range(1, len(str_1) + 1):
        for j in range(1, len(str_2) + 1):
            if str_1[i - 1] == str_2[j - 1]:
                tabel[i][j] = tabel[i - 1][j - 1] + 1
            else:
                tabel[i][j] = max(tabel[i - 1][j], tabel[i][j - 1])
    return tabel[len(str_1)][len(str_2)]


def max_sub_String(str_1, str_2):
    result = np.zeros([len(str_1) + 1, len(str_2) + 1])
    for i in range(1, len(str_1) + 1):
        for j in range(1, len(str_2) + 1):
            result[i][j] = result[i - 1][j - 1] + 1 if str_1[i - 1] == str_2[j - 1] else max(result[i - 1][j],
                                                                                             result[i][j - 1])
    return result[len(str_1)][len(str_2)]


#####################################################################################################################

###### Questions from leetcode.com Top 50 most common interview Questions ######

# You are given an integer array nums. You are initially positioned at the array's first index,
# and each element in the array represents your maximum jump length at that position.
# Return true if you can reach the last index, or false otherwise.
def canJump(nums: List[int]) -> bool:
    # greedy solution O(n)
    n = len(nums) - 1
    goal = n

    for i in range(n - 1, -1, -1):
        # if nums[n-1] + n-1 >= n we can get to the n elem from n-1
        if nums[i] + i >= goal:
            goal = i

    return goal == 0


# ********* DP Solution *********
# O(n^2)
def canJump1(nums: List[int]) -> bool:
    table = [False] * (len(nums))
    table[0] = True

    for i in range(1, len(table)):
        j = i - 1
        while j >= 0:
            if table[j] == True and nums[j] >= i - j:
                table[i] = True
                break
            else:
                j -= 1
    return table[-1]


#####################################################################################################################
# There is a robot on an m x n grid.
# The robot is initially located at the top-left corner (i.e., grid[0][0]).
# The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]).
# The robot can only move either down or right at any point in time.
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
def uniquePaths(m: int, n: int) -> int:
    # table = np.zeros([m, n])
    # # Base case of the first row and column
    # table[:1,:] = 1
    # table[:,:1] = 1
    table = []
    # base case first row all 1 (can only get there by moving right)
    table.append([1] * n)
    for row in range(1, m):
        # second base the first col is all 1 (can only get there by down)
        table.append([1])
        for col in range(1, n):
            # the number of paths is determaind by the sum
            # of paths leading to the above and left sub problem
            table[row].append(table[row - 1][col] + table[row][col - 1])
    return table[m - 1][n - 1]


#####################################################################################################################
# Given an integer array nums, return the length of the longest strictly increasing subsequence.
# A subsequence is a sequence that can be derived from an array by deleting some or no
# elements without changing the order of the remaining elements. For example,
# [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].
def lengthOfLIS(nums: List[int]) -> int:
    n = len(nums)
    table = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                table[i] = max(table[i], table[j] + 1)
    return max(table)


#####################################################################################################################
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
def maxProfit(prices: List[int]) -> int:
    results = [0] * len(prices)
    for i in range(1, len(prices)):
        results[i] = max(0, prices[i] - prices[i - 1] + results[i - 1])
    print(results)
    return max(results)


# improvements to the above solution.
# minprice = float('inf')
#  max_profit = 0
#  for i in range(len(prices)):
#      if(prices[i] < minprice):
#          minprice = prices[i]
#      elif (prices[i] - minprice > max_profit):
#          max_profit = prices[i] - minprice
#  return max_profit

#####################################################################################################################
# Given an integer array nums, find the contiguous subarray
# (containing at least one number) which has the largest sum and return its sum.
def maxSubArray(nums: List[int]) -> int:
    max_sub = nums[0]
    curr_sum = 0

    for num in nums:
        curr_sum += num
        if curr_sum > max_sub:
            max_sub = curr_sum
        if curr_sum < 0:
            curr_sum = 0
    return max_sub

#         max_sum = float('-inf')
#         result = []
#         for i in range(len(nums)):
#             max_sum = max(nums[i], max_sum + nums[i])
#             result.append(max_sum)
#         return max(result)

#####################################################################################################################
# You are a professional robber planning to rob houses along a street.
# Each house has a certain amount of money stashed,
# the only constraint stopping you from robbing each of them is that adjacent houses have security
# systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
# Given an integer array nums representing the amount of money of each house, return the
# maximum amount of money you can rob tonight without alerting the police.
def rob(nums: List[int]) -> int:
    include, skip = 0, 0

    for num in nums:
        curr_max = max(num + include, skip)
        include = skip
        skip = curr_max
    return max(skip, include)

#         table = [0] * len(nums)
#         if len(nums) == 1:
#             return nums[0]
#         if not nums:
#             return 0
#         table[0] = nums[0]
#         table[1] = nums[1]

#         for i in range(2, len(nums)):
#             table[i] = max(table[:i-1]) + nums[i]
#         return max(table[-1], table[-2])
