import random
from collections import Counter, defaultdict, deque
import re
from typing import List, Tuple, Dict, Deque, Optional
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
"""


############# Graphs and Trees  #############
class DirectedGraph():
    def __init__(self):
        self.adj_list = {}
        self.visited = defaultdict(lambda: False)
        self.vertices = []

    def add_vertex(self, vertex):
        self.adj_list[vertex] = []
        self.vertices.append(vertex)

    def add_edge(self, vertex1, vertex2):
        self.adj_list[vertex1].append(vertex2)


# This function checks if there is a path between two nodes in a Graph using BFS
# O(E + V) time and O(V) space
def route_between_bfs(graph, vertex_1, vertex_2):
    queue_1 = deque()
    queue_1.append(vertex_1)
    graph.visited[vertex_1] = True
    while (queue_1):
        curr_vertex = queue_1.popleft()
        for neighbor in graph.adj_list[curr_vertex]:
            if neighbor == vertex_2:
                return True
            if (graph.visited[neighbor] == False):
                queue_1.append(neighbor)
            graph.visited[neighbor] = True
    return False


# This function checks if there is a path between two nodes in a Graph using DFS
# O(E + V) time and O(V) space
def route_between_dfs(graph, vertex_1, vertex_2):
    graph.visited[vertex_1] = True
    if vertex_1 == vertex_2:  # base case
        return True
    for neighbor in graph.adj_list[vertex_1]:  # search in neighbors of vertex_1
        if graph.visited[neighbor] == False:
            return route_between_dfs(graph, neighbor, vertex_2)
    return False


class TreeNode():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return str(self.val)


# This function is building a BST from a sorted array, it uses the BST property and recursion.
# similar to binary search. The time complexity is O(n) and the space complexity is O(n)
def _arr_to_tree_helper(arr, l, r):
    if (l > r):
        return None
    mid = (r + l) // 2
    root = TreeNode(arr[mid])
    root.left = _arr_to_tree_helper(arr, l, mid - 1)
    root.right = _arr_to_tree_helper(arr, mid + 1, r)
    return root


def arr_to_tree(arr):
    _arr_to_tree_helper(arr, 0, len(arr))


# This function traverses a tree by level and returns a list of all the nodes in each level.
# O(n) time and O(n) space
def list_of_depths(root):
    bfs_queue = deque()
    level = 0
    # a list of lists to hold final result
    tree_levels = [[]]
    # we save for each vertex his depth the updating of level will be according to the level of parent
    bfs_queue.append((root, level))

    while (bfs_queue):
        curr_node, curr_level = bfs_queue.popleft()
        # if we are in a new level we add a new list to the list of lists
        if curr_level >= len(tree_levels):
            tree_levels.append([])
        tree_levels[curr_level].append(curr_node)  # inserting to the curr_node level list
        # we add the children of the curr_node to the queue
        if curr_node.left:
            bfs_queue.append((curr_node.left, curr_level + 1))
        if curr_node.right:
            bfs_queue.append((curr_node.right, curr_level + 1))
    return tree_levels


# def level(root):
#     if not root:
#         return 0
#     return max(level(root.left) + 1, level(root.right) + 1)

# def is_balanced(root):
#     if not root:
#         return True
#     res = is_balanced(root.left) and is_balanced(root.right)
#     left_height = height(root.left)
#     right_height = height(root.right)
#     if abs(left_height - right_height) > 1:
#         return False
#     else:
#         return res
#
# def height(root):
#     if not root:
#         return -1
#     left_height= height(root.left) + 1
#     right_height= height(root.right) + 1
#     return max(left_height, right_height)

# check if binary tree is balanced
# using bottom-up approach in order to avoid duplicate computation
# the float('inf') is used to represent an unbalanced tree error message which will be
# bubbled up to the top of the call stack existing the recursion call

# This is a helper function for the is_balanced function. It bubbles up the error message to the top of the call stack
# in case of an unbalanced tree. The solution is using the float('-inf') to represent an unbalanced tree.
# O(n) time and O(1) space
def tree_height(root):
    if not root:
        return -1
    left_height = tree_height(root.left)
    if left_height == float('-inf'):
        return float('-inf')

    right_height = tree_height(root.right)
    if right_height == float('-inf'):
        return float('-inf')

    height_diff = abs(right_height - left_height)
    return max(right_height, left_height) + 1 if height_diff < 2 else float('-inf')


def is_balanced(root):
    if not root:
        return True
    error_msg = float('-inf')
    return tree_height(root) != error_msg


#####################################################################################################################
# Question 4.5 Validate BST - first attempt using inorder traversal and check if the values are in order
# O(n) time and O(n) space
def is_sorted(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True


def is_bst_1(root):
    result = []

    def inorder_traversal(root):
        if not root:
            return
        inorder_traversal(root.left)
        result.append(root.data)
        inorder_traversal(root.right)

    inorder_traversal(root)
    return is_sorted(result)


# Question 4.5 Validate BST - second attempt using the BST property and recursion
# O(n) time and O(n) space
def _is_bst_helper(root, min, max):
    if not root:
        return True
    # the min value is the parent if we went right -> parent.right.val < parent.val
    # the max value is the parent if we went left -> parent.left.val > parent.val
    if (root.left.data < min or root.data > max):
        return False
    # recurse on the left and right subtrees, updating the min and max values accordingly
    return _is_bst_helper(root.left, min, root.data) and _is_bst_helper(root.right, root.data, max)


def is_bst(root):
    return _is_bst_helper(root, float('-inf'), float('inf'))


# find the min nude value in a BST
def min_node(root):
    curr = root
    while (curr.left):
        curr = curr.left
    return curr


# find the successor of a node in a BST
def successor(node: TreeNode):
    curr = node
    if not curr:
        return curr
    # First case if the node has a right subtree the successor is the minimum of that subtree
    if curr.right:
        return min_node(curr.right)
    # Second case the node successor is the first parent node that has the node as its left child
    else:
        while curr.parent:
            if curr == curr.parent.left:
                print(curr.parent.val)
                return curr.parent
            else:
                curr = curr.parent
        return node


# calculate the dis of a node from a root in a BST
def dis_from_root(node):
    if not node:
        return
    distance = 0
    while (node.parent):
        distance += 1
        node = node.parent
    return distance


# move a node up in the tree by swapping its value with its parent
def move_up_by(node, steps):
    while (node.parent and steps):
        node = node.parent
        steps -= 1
    return node


# This function is used to find the closest common ancestor of two nodes in a BST.
# The algorithm is based on the fact that the closest common ancestor is the lowest common ancestor of the two nodes
# that are farthest from each other. The solution is using dis_from_root to calculate the distance from the root of
# each node to the node we are looking for. Then we move up the tree by the difference between the two distances.
# O(n) time and O(1) space
def find_first_common_ancestor(node_1: TreeNode, node_2: TreeNode) -> TreeNode:
    if not node_1 or not node_2:
        return None
    first_depth = dis_from_root(node_1)
    second_depth = dis_from_root(node_2)
    deeper_node, shallower_node = (node_1, node_2) if first_depth > second_depth else (node_2, node_1)
    deeper_node = move_up_by(deeper_node, abs(first_depth - second_depth))
    if deeper_node == shallower_node:
        return deeper_node

    while ((shallower_node and deeper_node)):
        shallower_node = shallower_node.parent
        deeper_node = deeper_node.parent
        if shallower_node == deeper_node:
            return deeper_node
    return None


# This function checks if two trees are identical.
# The algorithm is to check if the two trees have the same values and the same children, using recursion.
# O(n) time and O(1) space
def equal_trees(T1, T2):
    if not T2 and not T1:
        return True
    if not T2 or not T1 or T1.val != T2.val:
        return False
    return equal_trees(T1.left, T2.left) and equal_trees(T1.right, T2.right)


# This function search whether T2 is a subtree of T1.
# Runtime is O(n) and space is O(1)
def search_sub_tree(T1: TreeNode, T2: TreeNode) -> bool:
    if (not T1 or not T2):
        return False

    if T1.val == T2.val:
        res = equal_trees(T1, T2)
        if res:
            return True

    return search_sub_tree(T1.right, T2) or search_sub_tree(T1.left, T2)


###### Questions from leetcode.com Top 50 most common interview Questions ######

#####################################################################################################################
# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center)
def isSymmetric(root: Optional[TreeNode]) -> bool:
    def helper(root_1, root_2):
        if (not root_1 and not root_2):
            return True
        if (not root_1 or not root_2):
            return False
        return (root_1.val == root_2.val) and helper(root_1.right, root_2.left) and helper(root_1.left, root_2.right)

    return helper(root, root)



#####################################################################################################################
# u are given two integer arrays nums1 and nums2, sorted in non-decreasing order,
# and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
# Merge nums1 and nums2 into a single array sorted in non-decreasing order.
# The final sorted array should not be returned by the function, but instead be stored inside the array nums1.
# To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged,
# and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
def merge(nums_1: List[int], m: int, nums_2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    last_index = n + m - 1
    n_counter = n - 1
    m_counter = m - 1
    while n_counter >= 0 and m_counter >= 0:
        if nums_1[m_counter] >= nums_2[n_counter]:
            nums_1[last_index] = nums_1[m_counter]
            m_counter -= 1
            last_index -= 1
        else:
            nums_1[last_index] = nums_2[n_counter]
            n_counter -= 1
            last_index -= 1
    print(nums_1)
    remainder_arr, num_elem = (nums_1, m_counter) if (m_counter > 0) else (nums_2, n_counter)
    print(remainder_arr, num_elem)
    for i in range(0, num_elem + 1):
        nums_1[i] = remainder_arr[i]
    print(nums_1)


# Given the root of a binary tree, return the zigzag level order traversal of its nodes' values.
# (i.e., from left to right, then right to left for the next level and alternate between).
def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    # Edge case
    if not root:
        return []
    # Initialize queue for bfs and final result
    res = []
    queue = deque()
    queue.append((root, 0))

    # BFS level order traverse
    while queue:
        curr_node, curr_level = queue.popleft()

        if curr_level >= len(res):
            res.append(deque())

        # Regular insertion no need for zigzag
        if curr_level % 2 == 0:
            res[curr_level].append(curr_node.val)
        # Odd level so we need to zigzag
        else:
            res[curr_level].appendleft(curr_node.val)

        if curr_node.left:
            queue.append((curr_node.left, curr_level + 1))
        if curr_node.right:
            queue.append((curr_node.right, curr_level + 1))
    return res


# Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree
# and inorder is the inorder traversal of the same tree, construct and return the binary tree.
def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    # Base case
    if not preorder or not inorder:
        return None

    root = TreeNode(preorder[0])
    # find the first occurrence of the root in the inorder lst
    ind = inorder.index(preorder[0])

    # everything to the left of curr val in inorder +
    # everything until ind in preorder -> left sub-tree
    root.left = buildTree(preorder[1:ind + 1], inorder[:ind])
    # everything else is to the right sub tree
    root.right = buildTree(preorder[ind + 1:], inorder[ind + 1:])

    return root


# Populate each next pointer to point to its next right node.
# If there is no next right node, the next pointer should be set to NULL.
# Initially, all next pointers are set to NULL.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect(root: 'Optional[Node]') -> 'Optional[Node]':
    if not root:
        return root
    queue = deque()
    queue.append((root, 0))
    # BFS level order traverse
    while queue:
        curr_node, curr_level = queue.popleft()

        if not queue or curr_level != queue[0][1]:
            curr_node.next = None
        else:
            curr_node.next = queue[0][0]

        if curr_node.left:
            queue.append((curr_node.left, curr_level + 1))
        if curr_node.right:
            queue.append((curr_node.right, curr_level + 1))
    return root


# Given the root of a binary search tree, and an integer k,
# return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    res = []
    global count

    def inorder(root, k):
        if not root:
            return None
        inorder(root.left, k)
        # res.append(root.val)
        count += 1
        if count == k:
            res.append(root.val)
        inorder(root.right, k)

    count = 0
    inorder(root, k)
    return res[0]


# create a class that represent a graph
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[0 for col in range(vertices)] for row in range(vertices)]

    def add_edge(self, u, v, w):
        self.graph[u][v] = w
        self.graph[v][u] = w

    def print_graph(self):
        for i in range(self.vertices):
            for j in range(self.vertices):
                print(self.graph[i][j], end=" ")
            print()


# implement dijkstra's algorithm
def dijkstra(graph, src):
    dist = [float("inf")] * graph.vertices
    dist[src] = 0
    visited = [False] * graph.vertices
    pq = []
    heapq.heappush(pq, (0, src))
    while pq:
        u = heapq.heappop(pq)[1]
        visited[u] = True
        for i in range(graph.vertices):
            if graph.graph[u][i] and not visited[i]:
                if dist[i] > dist[u] + graph.graph[u][i]:
                    dist[i] = dist[u] + graph.graph[u][i]
                    heapq.heappush(pq, (dist[i], i))
    return dist


def topological_sort(graph):
    in_degree = [0] * graph.vertices
    for i in range(graph.vertices):
        for j in range(graph.vertices):
            if graph.graph[i][j]:
                in_degree[j] += 1
    q = []
    for i in range(graph.vertices):
        if in_degree[i] == 0:
            q.append(i)
    result = []
    while q:
        u = q.pop(0)
        result.append(u)
        for i in range(graph.vertices):
            if graph.graph[u][i]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    q.append(i)
    return result


def prim(graph):
    dist = [float("inf")] * graph.vertices
    dist[0] = 0
    visited = [False] * graph.vertices
    pq = []
    heapq.heappush(pq, (0, 0))
    while pq:
        u = heapq.heappop(pq)[1]
        visited[u] = True
        for i in range(graph.vertices):
            if graph.graph[u][i] and not visited[i]:
                if dist[i] > graph.graph[u][i]:
                    dist[i] = graph.graph[u][i]
                    heapq.heappush(pq, (dist[i], i))
    return dist


# implement DFS algorithm
def dfs(graph, src):
    # create a deafultdict with graph.vertices as keys and values as False
    visited = defaultdict(lambda: False)
    visited[src] = True
    stack = []
    stack.append(src)
    while stack:
        u = stack.pop()
        print(u, end=" ")
        for i in range(graph.vertices):
            if graph.graph[u][i] and not visited[i]:
                stack.append(i)
                visited[i] = True


# implement BFS algorithm
def bfs(graph, src):
    visited = defaultdict(lambda: False)
    visited[src] = True
    queue = []
    queue.append(src)
    while queue:
        u = queue.pop(0)
        print(u, end=" ")
        for i in range(graph.vertices):
            if graph.graph[u][i] and not visited[i]:
                queue.append(i)
                visited[i] = True


# test the bfs function
def bfs_test():
    graph = Graph(5)
    graph.add_edge(0, 1, 5)
    graph.add_edge(0, 2, 3)
    graph.add_edge(1, 2, -1)
    graph.add_edge(1, 3, -2)
    graph.add_edge(2, 3, 4)
    graph.add_edge(3, 4, -3)
    graph.add_edge(4, 0, -4)
    graph.add_edge(4, 2, 1)
    graph.print_graph()
    print("BFS:")
    bfs(graph, 0)
    print()
    print("DFS:")
    dfs(graph, 0)


############## Some very basic tests ##############
# test list of depths
def test_list_of_depths():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)
    root.left.right.right = TreeNode(8)
    root.right.right.left = TreeNode(9)
    root.left.right.right.left = TreeNode(10)
    list_of_depths(root)


def test_route_between_dfs():
    graph = DirectedGraph()
    graph.add_vertex(1)
    graph.add_vertex(2)
    graph.add_vertex(3)
    graph.add_vertex(4)
    graph.add_vertex(5)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)


# test is_balanced
def test_is_balanced():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)
    root.left.right.right = TreeNode(8)
    root.right.right.left = TreeNode(9)
    root.left.right.right.left = TreeNode(10)
    assert is_balanced(root) == False
    root_2 = TreeNode(1)
    root_2.left = TreeNode(2)
    root_2.right = TreeNode(3)
    root_2.left.left = TreeNode(4)
    root_2.left.right = TreeNode(5)
    assert is_balanced(root_2) == True
    root_3 = TreeNode(1)
    root_3.left = TreeNode(2)
    assert is_balanced(root_3) == True
    root_4 = TreeNode(1)
    root_4.right = TreeNode(2)
    assert is_balanced(root_4) == True
    root_5 = TreeNode(1)
    root_5.left = TreeNode(2)
    root_5.right = TreeNode(3)
    root_5.left.left = TreeNode(4)
    root_5.left.right = TreeNode(5)
    assert is_balanced(root_5) == True
    root_5.right.right = TreeNode(6)
    root_5.right.right.right = TreeNode(7)
    assert is_balanced(root_5) == False


def test_find_common_ancestor():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)
    root.left.right.right = TreeNode(8)
    root.right.right.left = TreeNode(9)
    root.left.right.right.left = TreeNode(10)
    # define parent relationships
    root.left.parent = root
    root.right.parent = root
    root.left.left.parent = root.left
    root.left.right.parent = root.left
    root.right.left.parent = root.right
    root.right.right.parent = root.right
    root.left.right.right.parent = root.left.right
    root.right.right.left.parent = root.right.right
    root.left.right.right.left.parent = root.left.right.right
    assert find_first_common_ancestor(root.left, root.right).val == 1
    assert find_first_common_ancestor(root.left.left, root.right).val == 1
    assert find_first_common_ancestor(root.left.right.right.left, root.left.left).val == 2
    assert find_first_common_ancestor(root.left.right.right.left, root.right.right.left).val == 1
    assert find_first_common_ancestor(root.left.right.right.left, root.right.right.left).val == 1
