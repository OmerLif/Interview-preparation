import random
from collections import Counter, deque
import re
from typing import List, Tuple, Dict, Deque, Optional
from collections import defaultdict

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


########### Linked List page 92-94 #############
class Node:
    """
        This is an implementation of a linked list using a node class.
    """

    def __init__(self, value):
        self.val = value
        self.next = None

    def set_next(self, node) -> None:
        self.next = node

    def get_next(self):
        return self.next

    def get_value(self) -> int:
        return self.val


class LinkedList:
    """
        This is a possible implementation of Linked List class.
    """

    def __init__(self, node=None):
        self.head = node

    # add a node to the end of the list
    def add_node(self, value: int) -> None:
        new_node = Node(value)
        if (self.head == None):
            self.head = new_node
        else:
            current = self.head
            while (current.get_next() != None):
                current = current.get_next()
            current.set_next(new_node)

    # add a node to the beginning of the list
    def add_node_front(self, node: Node) -> None:
        if self.head == None:
            self.head = node
        else:
            node.next = self.head
            self.head = node

    # remove a node from the end of the list
    def remove_node(self, node: Node) -> None:
        if self.head == node:
            self.head = self.head.next
            del node
        else:
            curr = self.head
            while curr.next != node:
                curr = curr.next
            curr.next = curr.next.next

    # print the list
    def print_list(self) -> None:
        curr = self.head
        while curr != None:
            print(curr.val, end=" ")
            curr = curr.next

    # return the length of the list
    def len(self) -> int:
        curr = self.head
        count = 0
        while curr != None:
            count += 1
            curr = curr.next
        return count

    def Q_1_remove_duplicates_with_extra_buffer(self) -> Node:
        """
            This is a solution to the problem of removing duplicates from a linked list.
            The algorithm is O(n) time and O(n) space.
        :return:
        """
        curr = self.head
        counter_dict = defaultdict(lambda: 0)
        counter_dict[curr.val] = 1
        while (curr.next):
            counter_dict[curr.next.val] += 1
            if (counter_dict[curr.next.val] > 1):
                curr_next = curr.next
                curr.next = curr.next.next
                curr_next.next = None
                continue
            else:
                curr = curr.next
        # there is another way without using extra buffer, but it costs O(n^2) time
        # this solution is O(n) time but O(n) space for the counter_dict

    def sort_list(self) -> Node:
        """
            This is a solution to the problem of sorting a linked list, Using the bubble sort algorithm.
            The algorithm is O(n^2) time and O(1) space.
        :return:
        """
        if (self.head == None or self.head.next == None):
            return self.head
        length = 0
        curr = self.head
        while (curr):
            length += 1
            curr = curr.next
        i = 0
        curr = self.head
        while (i < length and curr):
            j = 0
            curr = self.head
            while (j < length - i - 1):
                if (curr.next and curr.next.val < curr.val):
                    # swapping L[j] with L[j+1] sorting by swapping the values
                    curr.next.val, curr.val = curr.val, curr.next.val
                curr = curr.next
                j += 1
            i += 1

    def find_middle_element(self) -> Node:
        """
            This is a solution to the problem of finding the middle element of a linked list.
            This algorithm is O(n) time and O(1) space.
        :return:
        """
        if (self.head == None):
            return None
        slow = self.head
        fast = self.head
        while (fast and fast.next):
            slow = slow.next
            fast = fast.next.next
        return slow


def middle(head: Node) -> Node:
    """
        This is a solution to the problem of finding the middle element of a linked list.
        It is used as a helper function for the merge sort algorithm.
        It is O(n) time and O(1) space.
    :param head: The first element of the list.
    :return:
    """
    fast = head
    slow = head
    while (fast.next and fast.next.next):
        slow = slow.next
        fast = fast.next.next
    return slow


def merge(left_node: Node, right_node: Node) -> Node:
    """
        This is a solution to the problem of merging two sorted linked lists.
        The algorithm is O(n) time and O(1) space.
    :param left_node:
    :param right_node:
    :return:
    """
    dummy_node = Node(0)
    curr = dummy_node
    left_dummy = left_node
    right_dummy = right_node
    while (left_dummy and right_dummy):
        if (left_dummy.val < right_dummy.val):
            dummy_node.next = left_dummy
            dummy_node = dummy_node.next
            left_dummy = left_dummy.next
        else:
            dummy_node.next = right_dummy
            dummy_node = dummy_node.next
            right_dummy = right_dummy.next
    dummy_node.next = right_dummy or left_dummy
    return curr.next


# This is a question from leetcode.com.
def merge_sort(head: Node) -> Node:
    """
        This is a solution to the problem of sorting a linked list using the merge sort algorithm.
        The algorithm is O(n log n) time and O(1) space.
    :param head:
    :return:
    """
    if (not head or not head.next):
        return head
    mid_node = middle(head)
    left_part = merge_sort(head)
    right_part = merge_sort(mid_node)
    return merge(left_part, right_part)


def Q_2_return_k_last(head: Node, K: int) -> Node:
    """
        This is a solution to the problem of finding the Kth element from the back of the list.
        For example, if the list is [1, 2, 3, 4, 5, 6, 7] and K is 3, then the answer is 4.
        and if k is 0, then the answer is 7.
        The algorithm is O(n) time and O(1) space.
    :param head: The first element of the list.
    :param K:
    :return:
    """
    fast = slow = head
    fast_gap = 0
    while (fast_gap < K):
        if (not fast):
            print(f"the length of the list is less than {K}!")
            return None
        fast = fast.next
        fast_gap += 1
    while (fast):
        fast = fast.next
        slow = slow.next
    return slow.val


def Q_5_sum_lists(node_1: Node, node_2: Node) -> Node:
    """
        This is a solution to the problem of adding two linked lists.
        This algorithm is based on the idea of adding the digits of the numbers from the back of the list.
        Same as the algorithm of adding two numbers. - long addition.
        The algorithm is O(n) time and O(1) space.
    :param node_1:
    :param node_2:
    :return:
    """
    curr = Node(0)
    new_head = curr
    reminder = 0
    while (node_1 and node_2):
        curr_sum = node_1.val + node_2.val + reminder
        reminder, digit = curr_sum // 10, curr_sum % 10
        curr.next = Node(digit)
        curr = curr.next
        node_1 = node_1.next
        node_2 = node_2.next
    longer_list = node_1 or node_2
    while (longer_list):
        curr_sum = longer_list.val + reminder
        reminder, digit = curr_sum // 10, curr_sum % 10
        curr.next = Node(digit)
        curr = curr.next
        longer_list = longer_list.next
    if reminder:
        curr.next = Node(reminder)
    return new_head.next


# This question is not solved in the book, but its from leetcode.com.
def reverse_and_clone(node: Node) -> Node:
    """
        This is a solution to the problem of reversing a linked list and cloning it.
        The algorithm is O(n) time and O(n) space.
    :param node:
    :return:
    """
    reverse_head = None
    while (node):
        temp_node = Node(node.val)
        temp_node.next = reverse_head
        reverse_head = temp_node
        node = node.next
    return reverse_head


def equal_lists(head: Node, tail: Node) -> bool:
    """
        This is a solution to the problem of checking if two linked lists are equal.
        The algorithm is O(n) time and O(1) space.
    :param head:
    :param tail:
    :return:
    """
    while (head and tail):
        if (head.val != tail.val):
            return False
        head = head.next
        tail = tail.next
    return True


def Q_6_palindrome(head: Node) -> bool:
    """
        This is a solution to the problem of checking if a linked list is a palindrome.
        The algorithm is O(n) time and O(1) space.
    :param head:
    :return:
    """
    # reverse and compare
    if (not head or not head.next):
        return True
    reversed_list = reverse_and_clone(head)
    res = equal_lists(head, reversed_list)
    return res


# This function is used to find the length of a linked list.
def lst_len(head: Node) -> int:
    len_ = 0
    while head:
        len_ += 1
        head = head.next
    return len_


# This question is in leetcode and in cracking the coding interview.
def Q_7_find_intersect(headA: Node, headB: Node) -> Node:
    """
        This is a solution to the problem of finding the intersection of two linked lists.
        The algorithm is O(n) time and O(1) space.
    :param headA:
    :param headB:
    :return:
    """
    while headA and headB:
        if headA == headB:
            return headA
        headA = headA.next
        headB = headB.next
    return


# This question is in leetcode and in cracking the coding interview. In cracking the coding interview,
# this question is called loop detection.
def Q_8_hasCycle(head: Optional[Node]) -> bool:
    """
        This is a solution to the problem of checking if a linked list has a cycle.
        The algorithm is O(n) time and O(1) space.
    :param head:
    :return:
    """
    slow_p = head
    fast_p = head
    while (slow_p and fast_p and fast_p.next):
        slow_p = slow_p.next
        fast_p = fast_p.next.next
        if slow_p == fast_p:
            return True
    return False


# Another Sol to find the intersection of two linked lists.
def getIntersectionNode(headA: Node, headB: Node) -> Optional[Node]:
    """
        This is a solution to the problem of finding the intersection of two linked lists.
        The algorithm is O(n) time and O(1) space.
    :param headA:
    :param headB:
    :return:
    """
    # Edge cases cover
    if not headA or not headB:
        return None

    # find the len of the two lists
    len_A = lst_len(headA)
    len_B = lst_len(headB)

    # profile the two lists
    diff_len = abs(len_A - len_B)
    max_len = max(len_A, len_B)
    longer_lst, shorter_lst = (headA, headB) if len_A > len_B else (headB, headA)

    # advance the longer lst so the two list
    # will be at equal dis from the end
    while diff_len > 0 and longer_lst:
        longer_lst = longer_lst.next
        diff_len -= 1
    # looking for intersect between the two lists
    return Q_7_find_intersect(longer_lst, shorter_lst)


# This question is from leetcode.com.
def oddEvenList(self, head: Optional[Node]) -> Optional[Node]:
    """
        This is a solution to the problem of alternating the nodes of a linked list.
        The algorithm is O(n) time and O(1) space.
    :param self:
    :param head:
    :return:
    """
    lst_len = 0
    tmp_head = head
    tail = head
    while tmp_head:
        lst_len += 1
        tail = tmp_head
        tmp_head = tmp_head.next

    # Special edge case
    if lst_len <= 2:
        return head

    index = 1
    perv = None
    curr = head

    while index <= lst_len:
        # if we hit an even index we remove that node,
        # and append it to the tail of the list

        if index % 2 == 0:
            nxt = curr.next
            perv.next = curr.next
            curr.next = None
            tail.next = curr
            tail = tail.next
            curr = nxt
            index += 1

        else:
            perv = curr
            curr = curr.next
            index += 1
    return head


def reverse_linked_list(head: Node) -> Node:
    """
        This is a solution to the problem of reversing a linked list.
        The algorithm is O(n) time and O(1) space.
    :param head:
    :return:
    """
    prev = None
    curr = head
    while (curr):
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev


def reverse_list(head: Node) -> Node:
    """
        This is a solution to the problem of reversing a linked list.
        The algorithm is O(n) time and O(1) space.
    :param head:
    :return:
    """
    if not head or not head.next:
        return head
    perv = None
    curr = head
    while (curr.next):
        next = curr.next
        curr.next = perv
        perv = curr
        curr = next
    return perv


# test find_k_elem function
def test_find_k_elem():
    pass
