import random
import re
from typing import List, Tuple, Dict, Deque
from math import inf
from collections import Counter, defaultdict
from functools import reduce

"""
    this is a solution for many of the problems in cracking the code interview and leetcode 
    Top interview Questions.
    I will be using the python 3.6.4 version of python.
    I will be using the following libraries:
    - collections
    - re
    - typing
    - copy
    - functools
    - math
    - random
"""


################### Array and strings page 90 -  ########################


def Q_1_all_unique_str(my_str: str) -> bool:
    """
            Q.1 This function will return true if all the characters in the string are unique.
            O(1) time and O(1) space.
        :param my_str: current string
        :return: boolean if all the characters in the string are unique
    """
    if (len(my_str) > 256):
        return False
    letters_lst = [0] * 256
    for char in my_str:
        letters_lst[ord(char)] += 1
        if (letters_lst[ord(char)]) > 1:
            return False
    return True


def Q_2_check_if_per(str_1: str, str_2: str) -> bool:
    """
        Q.2 This function will return true if str_1 is a permutation of str_2.
        O(n) time and O(1) space.
    :param str_1: First string
    :param str_2: Second string
    :return: true if str_1 is a permutation of str_2
    """
    return Counter(str_1) == Counter(str_2)


def Q_3_replace_spaces(sen: str) -> str:
    """
        Q.3 This function will replace all the spaces in a string with %20.
        O(n) time and O(1) space.
        re.sub in general take O(2^m + n), when m=len(regex) and n=len(sen).
    :param sen: The string to replace spaces in.
    :return: The string with all the spaces replaced with %20
    """
    return re.sub('\s+', '%20', sen)


def Q_4_check_if_palindrome(word: str) -> bool:
    """
        Q.4 This function will return true if the word is a palindrome.
        O(n) time and O(1) space.
    :param word: The word to check
    :return: True if the word is a palindrome else false
    """
    word = word.replace(" ", "").lower()
    counter_dict = Counter(word)
    odd_count = 0
    for key, val in counter_dict.items():
        if (counter_dict[key] % 2 != 0):
            odd_count += 1
            if odd_count > 1:
                return False
    return True


def one_edit_replacement(str_1: str, str_2: str) -> bool:
    """
        This function is a helper function for Q_5_one_edit_strings.
        It will return true if the strings are one edit away from each other.
        O(n) time and O(1) space.
    :param str_1: The first string
    :param str_2: The second string
    :return:
    """
    had_replaced = False
    for i in range(len(str_1)):
        if (str_2[i] != str_1[i]):
            if (had_replaced):
                return False
        else:
            had_replaced = True
    return True


def one_edit_insert(long_str: str, short_str: str) -> bool:
    """
        This function is a helper function for Q_5_one_edit_strings.
        It will return true if the strings are one edit away from each other.
        O(n) time and O(1) space.
    :param long_str:
    :param short_str:
    :return:
    """
    long_index = short_index = 0
    while (long_index < len(long_str) and short_index < len(short_str)):
        if (long_str[long_index] != short_str[short_index]):
            if (long_index != short_index):
                return False
            else:
                long_index += 1
        else:
            long_index += 1
            short_index += 1
    return True


def Q_5_one_edit_strings(str_1: str, str_2: str) -> bool:
    """
        Q.5 This function will return true if the strings are one edit away from each other.
        It will return false if the strings are not one edit away from each other.
        O(n) time and O(1) space.
    :param str_1:
    :param str_2:
    :return:
    """
    long_str, short_str = (str_1, str_2) if (len(str_1) > len(str_2)) else (str_2, str_1)
    if (len(long_str) == len(short_str)):
        return one_edit_replacement(str_1, str_2)
    else:
        return one_edit_insert(long_str, short_str)


def Q6_compressed_string(string):
    """
        Q.6 This function will return the compressed string.
        Example: aaabbcccc -> a3b2c4
        O(n) time and O(1) space.
    :param string: The string to compress
    :return:
    """
    curr_counter = 0
    got_pressed = False
    perv = result_str = string[0]
    for i in range(1, len(string)):
        if (string[i] == perv):
            curr_counter += 1
            if curr_counter > 1:
                got_pressed = True
        else:
            result_str += str(curr_counter)
            curr_counter = 0
            result_str += perv
            perv = string[i]
    if (got_pressed):
        return result_str
    return string


# Define new object type - Matrix
Matrix = List[List[int]]
NO_PASS = -1


def Q_7_rotate_matrix_90(mat: Matrix) -> Matrix:
    """
        Q.7 This function rotate a matrix by 90 degrees in Place.
        O(n*m) time and O(1) Space.
    :param matrix:
    :return:
    """

    # naive solution
    # result_mat = np.zeros([len(mat), len(mat[0])])
    # for i, row in enumerate(mat):
    #     for j, cur_val in enumerate(row):
    #         result_mat[j][len(mat[0]) - i - 1] = cur_val
    # return result_mat
    left = top = 0
    right = bottom = len(mat) - 1
    while left < right and top < bottom:
        for i in range(right - left):
            temp = mat[top][left + i]

            # Move left bottom corner to the top left corner
            mat[top][left + i] = mat[bottom - i][left]

            # Move right bottom corner to the left bottom corner
            mat[bottom - i][left] = mat[bottom][right - i]

            # Move top right corner to bottom right corner
            mat[bottom][right - i] = mat[top + i][right]

            # Final shifting pase taking the top right corner and fills it with the temp var
            mat[top + i][right] = temp
        bottom -= 1
        top += 1
        left += 1
        right -= 1
        print("After rotate: ", mat)
    return mat


def is_substring(str_1, str_2):
    """
        This function will return true if str_1 is a substring of str_2.
    :param str_1:
    :param str_2:
    :return:
    """
    return re.search(str_1, str_2) != None


def Q_9_string_rotation(str_1, str_2) -> bool:
    if (len(str_1) != len(str_2)):
        return False
    if (is_substring(str_1 + str_1, str_2)):
        return True
    return False


###### Questions from leetcode.com Top 50 most common interview Questions ######

def is_anagram(s1, s2):
    if len(s1) != len(s2):
        return False
    s1 = s1.lower()
    s2 = s2.lower()
    s1_dict = {}
    s2_dict = {}
    for i in s1:
        if i in s1_dict:
            s1_dict[i] += 1
        else:
            s1_dict[i] = 1
    for i in s2:
        if i in s2_dict:
            s2_dict[i] += 1
        else:
            s2_dict[i] = 1
    if s1_dict == s2_dict:
        return True
    else:
        return False


def anagram_test():
    print(is_anagram('anagram', 'nagaram'))
    print(is_anagram('rat', 'car'))
    print(is_anagram('a', 'a'))
    print(is_anagram('', ''))
    print(is_anagram('a', 'b'))


# Given a list of integers, arrange the
# numbers such that odd numbers come before even numbers.
# Follow-up: solve it in o(n) time and o(1) space
def odd_even_sort(arr):
    if len(arr) == 0:
        return arr
    odd_idx = 0
    even_idx = len(arr) - 1
    while odd_idx < even_idx:
        if arr[odd_idx] % 2 == 1:
            odd_idx += 1
        else:
            arr[odd_idx], arr[even_idx] = arr[even_idx], arr[odd_idx]
            even_idx -= 1
    return arr


def odd_even_sort_test():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(odd_even_sort(lst))
    lst_1 = random.sample(range(100), 10)
    print(odd_even_sort(lst_1))


# Given 2 strings, merge the two strings such that the duplicates are removed and the order should be maintained.
def merge_strings(s1, s2):
    if len(s1) == 0:
        return s2
    if len(s2) == 0:
        return s1
    s1_dict = {}
    for c in s2:
        if c in s1_dict:
            s1_dict[c] += 1


# test the merge_strings function
def merge_strings_test():
    s1 = 'abcdefghijklmnopqrstuvwxyz'
    s2 = 'abcdefghijklmnopqrstuvwxyz'
    print(merge_strings(s1, s2))
    s1 = 'dogandcat'
    s2 = 'catanddog'
    print(merge_strings(s1, s2))


# This function recives a string representing a number in Roman Numerals and returns the integer value of the number
def roman_to_int(s):
    # if len(s) == 0:
    #     return 0
    # s_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    # prev = 0
    # total = 0
    # for i in s:
    #     if s_dict[i] > prev:
    #         total += s_dict[i] - 2 * prev
    #
    #     else:
    #         total += s_dict[i]
    #     prev = s_dict[i]
    # return total
    roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    string_length = len(s)
    roman_integer = 0
    skip_next_char = False
    if (string_length < 1 or string_length > 15):
        print("This is an invalid string!")
    if any(c not in roman_dict for c in s):
        print("string must contains only Roman letters!")
    for i, sign in enumerate(s):
        if skip_next_char:
            skip_next_char = False
            continue
        if (i < string_length - 1):
            if roman_dict[sign] < roman_dict[s[i + 1]]:
                # special case were we need to subtract
                roman_integer += (roman_dict[s[i + 1]] - roman_dict[sign])
                skip_next_char = True
            else:
                roman_integer += roman_dict[sign]
                skip_next_char = False
        else:
            roman_integer += roman_dict[sign]
    return roman_integer


# write a function that convert a number to a string in Roman Numerals
def int_to_roman(n):
    roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    roman_string = ''
    for k, v in roman_dict.items():
        if n >= v:
            roman_string += k
            n -= v
    return roman_string


def removeDuplicates(nums) -> int:
    if len(nums) == 0:
        return 0
    i = 0
    for j in range(len(nums)):
        if nums[i] != nums[j]:
            i += 1
            nums[i] = nums[j]
    return i + 1


def rotate_array(nums, k):
    if len(nums) == 0:
        return 0
    for i in range(k):
        nums.append(nums[0])
        del nums[0]
    return nums


# write a function that rotate an array to the left by k steps
def rotate_array_left(nums, k):
    if len(nums) == 0:
        return 0
    for i in range(k):
        nums.insert(0, nums[-1])
        del nums[-1]

    return nums


# This function moves all the zero's to the end of the array
def clear_zeroes(nums):
    last_place = 0
    for i in range(len(nums)):
        if (nums[i]):
            nums[i], nums[last_place] = nums[last_place], nums[i]
            last_place += 1
    return nums


#####################################################################################################################
# Given an array, rotate the array to the right by k steps, where k is non-negative.
# This solution is O(kn) time and O(1) space, can use deque to solve in O(k) time and O(n) space.
# We consider inserting an element to the end of the array as O(n) operation.
def rotate(nums: List[int], k: int) -> None:
    for i in range(k):
        nums.insert(0, nums.pop())


#####################################################################################################################
# Given an integer array nums, return true if any value appears at least twice in the array,
# and return false if every element is distinct.
# This solution is O(nlog(n)) time and O(n) place
# below you can see solution using hash table in O(n) time and O(n) space
def containsDuplicate(nums: List[int]) -> bool:
    nums = sorted(nums)  # in place sorting
    perv = nums[0]
    for i in range(1, len(nums)):
        if (nums[i] == perv):
            return True
        perv = nums[i]
    return False


# counter_dict = Counter(nums)
# return any(count > 1 for count in counter_dict.values()) has better avg case solution than sorting
# any(nums.count(num) > 1 for num in nums)


####################################################################################################################
# You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
# On each day, you may decide to buy and/or sell the stock.
# You can only hold at most one share of the stock at any time.
# However, you can buy it then immediately sell it on the same day.
# Find and return the maximum profit you can achieve.
# This solution is O(n) time and O(1) space
def maxProfit(prices: List[int]) -> int:
    max_profit = 0
    for index in range(1, len(prices)):
        if (prices[index] > prices[index - 1]):
            max_profit += (prices[index] - prices[index - 1])
    return max_profit


####################################################################################################################
# You are given a large integer represented as an integer array digits,
# where each digits[i] is the ith digit of the integer.
# The digits are ordered from most significant to least significant in left-to-right order.
# The large integer does not contain any leading 0's.
# Increment the large integer by one and return the resulting array of digits.
# This solution is O(n) time and O(1) space
def plusOne(digits: List[int]) -> List[int]:
    return list(str(int(reduce(lambda a, b: str(a) + str(b), digits)) + 1))


####################################################################################################################
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.
# This function is O(n) time and O(n) space
def twoSum(nums: List[int], target: int) -> List[int]:
    num_index_dict = dict()
    for i, num in enumerate(nums):
        if (num not in num_index_dict):
            num_index_dict[num] = i
        else:
            if (num * 2 == target):
                return [num_index_dict[num], i]
    for key, value in num_index_dict.items():
        if ((target - key) in num_index_dict and value != num_index_dict[target - key]):
            return [value, num_index_dict[target - key]]


###################################################################################################################
# You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
# You have to rotate the image in-place,
# which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
# This function is O(n^2) time and O(1) space
def rotate(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    l, r = 0, len(matrix) - 1
    while (l < r):
        for i in range(r - l):
            t, b = l, r
            # rotateing in -90 degrees
            top_left = matrix[t][l + i]

            # replacing top left with bottom left
            matrix[t][l + i] = matrix[b - i][l]

            # replacing bottom left with bottom right
            matrix[b - i][l] = matrix[b][r - i]

            # replacing bottom right with top right
            matrix[b][r - i] = matrix[t + i][r]

            # replacing top right with top left
            matrix[t + i][r] = top_left
        l += 1
        r -= 1


# Write a function to find the longest common prefix string amongst an array of strings.
# If there is no common prefix, return an empty string ""
# This solution is O(n) time and O(1) space
def longestCommonPrefix(strs: List[str]) -> str:
    shortest_string = min(strs, key=lambda s: len(s))
    res =""
    for i in range(len(shortest_string)):
        if any(s[i] != shortest_string[i] for s in strs):
            return res
        else:
            res += shortest_string[i]
    return res



###################################################################################################################
# Given an integer array nums, return true if there exists a triple of indices (i, j, k)
# such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.
# This function is O(n) time and O(1) space
def increasingTriplet(nums: List[int]) -> bool:
    first, second = inf, inf
    for third in nums:
        if third <= first:
            first = third
        elif third <= second:
            second = third
        else:
            return True
    return False


###################################################################################################################
# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]]
# such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# This function is O(n^2) time and O(n) space
def threeSum(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()
    for i, item in enumerate(nums):
        if i > 0 and item == nums[i - 1]:
            continue
        l = i + 1
        r = len(nums) - 1
        if (nums[i] > 0):
            break
        while l < r:
            three_sum = nums[l] + nums[r] + nums[i]
            if three_sum > 0:
                r -= 1
            elif three_sum < 0:
                l += 1
            else:
                res.append([nums[l], nums[r], nums[i]])
                l += 1
                # making sure we wont create duplicated threepels
                while nums[l] == nums[l - 1] and l < r:
                    l += 1

    return res


###################################################################################################################
# Given an m x n integer matrix, if an element is 0, set its entire row and column to 0's.
# You must do it in place.
# This solution is O(m*n) time and O(1) space

# A function that check if a matrix col contained zero's
def zero_in_col(self, matrix: List[List[int]], index) -> bool:
    for i in range(len(matrix)):
        if matrix[i][index] == 0:
            return True
    return False


# A function that color a matrix col with 0
def color_matrix_col(matrix: List[List[int]], index) -> None:
    for i in range(len(matrix)):
        matrix[i][index] = 0


def setZeroes(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    zero_at_row = 0 in matrix[0]
    zero_at_col = zero_in_col(matrix, 0)

    for row in range(1, len(matrix)):
        for col in range(1, len(matrix[0])):
            # Mark down all rows and cols that are going to be set to zero
            if matrix[row][col] == 0:
                matrix[row][0] = 0
                matrix[0][col] = 0
    # Color all the necessary rows
    for i in range(1, len(matrix)):
        if matrix[i][0] == 0:
            matrix[i] = [0] * len(matrix[i])

    # Color all the necessary rows
    for j in range(1, len(matrix[0])):
        if matrix[0][j] == 0:
            color_matrix_col(matrix, j)
    # color the first row and col if necessary
    if zero_at_row:
        matrix[0] = [0] * len(matrix[0])
    if zero_at_col:
        color_matrix_col(matrix, 0)


###################################################################################################################
# Given an array of strings strs, group the anagrams together. You can return the answer in any order.
# An Anagram is a word or phrase formed by rearranging
# the letters of a different word or phrase, typically using all the original letters exactly once.
# This function is O(n) time and O(n) space`
def groupAnagrams(strs: List[str]) -> List[List[str]]:
    counter_dict = defaultdict(list)
    for anagram in strs:
        counter_dict[''.join(sorted(anagram))].append(anagram)
    return [val for key, val in counter_dict.items()]


###################################################################################################################
# Given a string s, find the length of the longest substring without repeating characters.
# This function is O(n) time and O(n) space
def lengthOfLongestSubstring(s: str) -> int:
    l = curr_max = 0
    char_set = set()

    for r in range(len(s)):
        # removing characters from a the beginning of a string
        while s[r] in char_set:
            char_set.remove(s[l])
            l += 1
        # s[r] not in the set so we could add it to the set
        char_set.add(s[r])
        # r -l +1 is the length of the curr substring
        curr_max = max(r - l + 1, curr_max)
    return curr_max


###################################################################################################################
# Given a string s, return the longest palindromic substring in s.
# This function is O(n) time and O(n) space
def longestPalindrome(s: str) -> str:
    # Sliding window technique solution
    n = len(s)
    if n <= 1:
        return s

    curr_len = res_len = 0
    res = ""

    # Choose a char to be the middle of the polindrom
    for i in range(n):
        # window pointers - the odd length polindrom
        l, r = i, i
        while l >= 0 and r < n and s[l] == s[r]:
            curr_len = r - l + 1
            if curr_len > res_len:
                res = s[l:r + 1]
                res_len = curr_len
            l -= 1
            r += 1

        # window pointers - the even length polindrom
        l, r = i, i + 1
        while l >= 0 and r < n and s[l] == s[r]:
            curr_len = r - l + 1
            if curr_len > res_len:
                res = s[l:r + 1]
                res_len = curr_len
            l -= 1
            r += 1

    return res


# test the roman_to_int function
def test_roman_to_int():
    print(roman_to_int('MCMXCIV'))
    print(roman_to_int('MCMLIV'))
    print(roman_to_int('MCMLXVIII'))
    print(roman_to_int('MCMLXXXIX'))
    print(roman_to_int('III'))
    print(roman_to_int('IV'))
    print(roman_to_int('IX'))
    print(roman_to_int('X'))
    print(roman_to_int('XI'))
    print(roman_to_int('XV'))


# test Q_1_all_unique_str
def test_Q_1_all_unique_str():
    print(Q_1_all_unique_str('abcdefg'))
    print(Q_1_all_unique_str('abcdefg' * 10))
    print(Q_1_all_unique_str('abcdefgg'))
    print(Q_1_all_unique_str('abccdefg'))


# create a test for Q_2_check_if_per
def test_Q_2_check_if_per():
    print(Q_2_check_if_per('abcdefg', 'abcdefg'))
    print(Q_2_check_if_per('abcdefg', 'abcdefgg'))
    print(Q_2_check_if_per('abcdefg', 'abccdefg'))
    print(Q_2_check_if_per('abcdefg', 'abcd'))
    print(Q_2_check_if_per('abc', 'bca'))


# test Q_4_check_if_palindrome
def test_Q_4_check_if_palindrome():
    print(Q_4_check_if_palindrome('abcdefg'))
    print(Q_4_check_if_palindrome('tact coa'))
    print(Q_4_check_if_palindrome('tactoa'))
    print(Q_4_check_if_palindrome('tact oa'))


# test Q_5_one_edit_strings
def test_Q_5_one_edit_strings():
    assert Q_5_one_edit_strings('abcdefg', 'abcdefg') == True
    assert Q_5_one_edit_strings('abcdefg', 'abcdefgg') == False
    assert Q_5_one_edit_strings('abcdefg', 'abccdefg') == True
    assert Q_5_one_edit_strings('ab', 'abcd') == False
    assert Q_5_one_edit_strings('a', 'c') == True
    assert Q_5_one_edit_strings('a', 'bb') == False


# test Q6_compressed_string
def test_Q6_compressed_string():
    assert Q6_compressed_string('aaabbcccc') == 'a3b2c4'
    assert Q6_compressed_string('a%%bbb ') == 'a%3b3'
    assert Q6_compressed_string('a%%bbb ' * 10) == 'a%3b3' * 10
    assert Q6_compressed_string('a%%bbb ' * 100) == 'a%3b3' * 100
    assert Q6_compressed_string("") == ''  # empty string


# test Q_7_rotate_matrix_90
def test_Q_7_rotate_matrix_90():
    # assert 2X2 matrix
    mat = [[1, 2], [3, 4]]
    assert Q_7_rotate_matrix_90(mat) == [[3, 1], [4, 2]]
    assert Q_7_rotate_matrix_90([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    assert Q_7_rotate_matrix_90([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]) == [[13, 9, 5, 1],
                                                                                                     [14, 10, 6, 2],
                                                                                                     [15, 11, 7, 3],
                                                                                                     [16, 12, 8, 4]]


# test Q_9_string_rotation
def test_Q_9_string_rotation():
    assert Q_9_string_rotation('abcdefg', 'abcdefg') == True
    assert Q_9_string_rotation('abcdefg', 'abcdefgg') == False
    assert Q_9_string_rotation('abcdefg', 'abccdefg') == True
    assert Q_9_string_rotation('abcdefg', 'abcd') == False
    assert Q_9_string_rotation('abc', 'bca') == True
    assert Q_9_string_rotation('a', 'bca') == False
    assert Q_9_string_rotation('a', 'b') == False
    assert Q_9_string_rotation('a', 'a') == True
    assert Q_9_string_rotation('a', '') == False
    assert Q_9_string_rotation('', 'a') == False
    assert Q_9_string_rotation('', '') == True


# test the clear_zeroes function
def clear_zeroes_test():
    print(clear_zeroes([0, 1, 0, 3, 12]))
    print(clear_zeroes([0, 0, 1, 3, 12]))
    print(clear_zeroes([1, 0, 0, 3, 12]))
    print(clear_zeroes([1, 0, 0, 0, 0]))
    print(clear_zeroes([0, 0, 0, 0, 0]))
    print(clear_zeroes([0, 0, 0, 0, 0, 0]))
    print(clear_zeroes([0, 0, 0, 12, 0, 0]))
    print(clear_zeroes([0, 0, 0, 0, 0, 0, 1]))


# test the removeDuplicates function
def removeDuplicates_test():
    print(removeDuplicates([1, 2, 3, 3, 3, 4, 4, 4, 5]))
    print(removeDuplicates([1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5]))
    print(removeDuplicates([1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5]))


# test the int_to_roman function
def test_int_to_roman():
    print(int_to_roman(1))
    print(int_to_roman(2))
    print(int_to_roman(3))
    print(int_to_roman(4))
    print(int_to_roman(5))
    print(int_to_roman(6))
    print(int_to_roman(7))
    print(int_to_roman(8))
    print(int_to_roman(9))
    print(int_to_roman(10))
    print(int_to_roman(11))
    print(int_to_roman(12))
    print(int_to_roman(13))
    print(int_to_roman(14))
    print(int_to_roman(15))
    print(int_to_roman(16))
    print(int_to_roman(17))
    print(int_to_roman(18))
    print(int_to_roman(19))
    print(int_to_roman(1994))
