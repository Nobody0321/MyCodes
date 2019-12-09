# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root is None:
            return True
        elif not (root.left or root.right):
            return True
        else:
            return self.dfs(root.left, root.right)
        

    def dfs(self, leftnode, rightnode):
        if not (leftnode or rightnode):
            return True
        elif ((not leftnode) and rightnode) or ((not rightnode) and leftnode) or (leftnode.val != rightnode.val):
            return False
        else:
            return (self.dfs(leftnode.left, rightnode.right) and self.dfs(rightnode.left, leftnode.right))