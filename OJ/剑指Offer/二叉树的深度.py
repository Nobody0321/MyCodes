# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.length = 0

    def TreeDepth(self, pRoot):
        self.leng(pRoot, 0)
        return self.length

    def leng(self, node, l):
        if node:
            l += 1
            self.leng(node.left, l)
            self.leng(node.right, l)
        else:
            if self.length < l:
                self.length = l