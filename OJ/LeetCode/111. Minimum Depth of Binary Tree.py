# 一棵树的最小高度，是其左右子树最小高度的最小值，若有一个子树为空，那高度就为0
# 难点在当一个子树为空时，该节点的最小高度等于另一个子树的最小高度。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        elif root.left == None:
            return 1 + self.minDepth(root.right)
        elif root.right == None:
            return 1 + self.minDepth(root.left)
        else:
            return 1 + min(self.minDepth(root.left), self.minDepth(root.right))