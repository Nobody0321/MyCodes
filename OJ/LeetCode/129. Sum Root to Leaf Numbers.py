# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumNumbers(self, root):
        self.pathSum = 0
        self.pathSums(root, 0)
        return pathSum
    
    # 思路：定义一个函数，每次遇到一个新的节点，就把之前的数字乘以10再加上当前值，
    # 对我难点在于终止条件，究竟是在一个空节点终止（这样会导致重复叶子节点的路径被加两次）
    def pathSums(self, node, path):
        if node:
            path = 10 * path + node.val
        if node.left or node.right:
            if node.left:
                self.pathSums(node.left, path)
            if node.right:
                self.pathSums(node.right, path)
        else:
            self.pathSum += path