# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root, target):
        self.result = []
        if root != None:
            self.backTrack(root, target, [])
        return self.result

    def backTrack(self, node, target, path):
        t = node.val
        if (node.left == None and node.right == None) and target == t:
            # 到达叶子节点且已经满足target
            self.result.append(path+[t]) 
            return

        else:
            if node.left:
                self.backTrack(node.left, target - t, path+[t])
            if node.right:
                self.backTrack(node.right, target - t, path+[t])