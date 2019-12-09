# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        # 什么是对称：对于根节点，其左右子树关于根节点对称，且左右子树也是对称的
        if pRoot == None:
            # 空树也是对称的
            return True 
        else:
            return self.symmetrical(pRoot.left, pRoot.right)

    def symmetrical(self, node1, node2):
        if node1 == None and node2 == None:
            # 全为空也是对称
            return True
        elif (not node1) or (not node2):
            # 不全为空肯定不对称
            return False
        else:
            return (node1.val == node2.val) and self.symmetrical(node1.left, node2. right) and self.symmetrical(node1.right, node2.left)