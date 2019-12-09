# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root):
        result = []
        def pt(node, res):
            if node :
                res.append(node.val)
                pt(node.left, res)
                pt(node.right, res)
        pt(root,result)
        return result