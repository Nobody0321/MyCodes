# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        return self.dfs(p,q)


    def dfs(self, p, q):
        if not (p or q):
            return True
        elif not (p and q):
            return False
        else:
            if p.val == q.val:
                return (self.dfs(p.left, q.left) and self.dfs(p.right, q.right))
            else:
                return False