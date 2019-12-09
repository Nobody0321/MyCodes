# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 使用一个flag 标志层的结束，只有最后一层所有子节点都为空才会结束遍历
# 一系列自然数相乘，有一个是0就是0,所有都为正才是正
class Solution:
    def levelOrder(self, root):
        if root == None:
            return []

        levels = [[root]]  # 按层存点
        level = []  # 存一层的点
        ret = [[root.val]]  # 按层存值
        t = []  # 存一层的值
        while True:
            for node in levels[-1]:
                if node.left or node.right:
                    if node.left:
                        level.append(node.left)
                        t.append(node.left.val)
                    if node.right:
                        level.append(node.right)
                        t.append(node.right.val)
            if t == []:
                # 如果最后是[] 说明上一层是最后一层
                return ret
            levels.append(level)
            ret.append(t)
            t = []
            level = []