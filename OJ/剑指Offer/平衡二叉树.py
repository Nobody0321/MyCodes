# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        if pRoot == None:
            return True
        return abs(self.depth(pRoot.left) - self.depth(pRoot.right)) <= 1 \
            and self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
    
    def depth(self, node):
        if node == None:
            return 0
        else:
            return 1 + max(self.depth(node.left), self.depth(node.right))