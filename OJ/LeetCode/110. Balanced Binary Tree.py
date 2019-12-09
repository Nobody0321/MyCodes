# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        
        leftheight = self.getDepth(root.left) if root.left else 0 
        rightheight = self.getDepth(root.right) if root.right else 0  
        
        if abs(leftheight - rightheight) <=1:
            return self.isBalanced(root.left) and self.isBalanced(root.right) 
        else:
            return False
        
    def getDepth(self, treeNode):
        if treeNode == None:
            return 0
        else:
            return 1 + max(self.getDepth(treeNode.left), self.getDepth(treeNode.right))
