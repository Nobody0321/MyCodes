# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        results = []
        self.dfs(results, root, 1)
        return results[::-1]

    def dfs(self, results, TreeNode, depth):        
        if TreeNode == None:
            return
        if len(results) < depth:
            results.append([])
        results[depth-1].append(TreeNode.val)
        self.dfs(results, TreeNode.left, depth + 1)
        self.dfs(results, TreeNode.right, depth + 1)