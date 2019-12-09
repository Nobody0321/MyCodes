# 这是一个有就行的问题，所以可以递归查找到叶子节点，只要有一条路径满足就返回True

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        preSum = 0
        return self.Nodesum(sum, preSum, root)
        
    
    def Nodesum(self, sum, preSum, TreeNode):
        if TreeNode == None:
            # maybe TreeNode has a sibling left/right node, so the search cannot stop
            return False
        if TreeNode.left == None and TreeNode.right == None:
            # so TreeNode is a leaf node
            return True if preSum + TreeNode.val == sum else False
        else:
            return self.Nodesum(sum, preSum + TreeNode.val, TreeNode.left) or self.Nodesum(sum, preSum+TreeNode.val, TreeNode.right)
