# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        # 思路一：奇数行原序，偶数行逆序
        node_levels = [[root]]
        node_level = []
        val_levels = [[root.val]]
        val_level = []
        level = 2  # 记录层数，node_levels已经不为空，所以是从第二层开始
        while True:
            for node in node_levels[-1]:
                if node.left or node.right:
                    if node.left:
                        val_level.append(node.left.val)
                        node_level.append(node.left)
                    if node.right:
                        val_level.append(node.right.val)
                        node_level.append(node.right)
            if node_level == []:
                # 说明已经当前存储了的是最后一层
                return val_levels
            if level % 2 == 1:
                val_level.reverse()
            val_levels.append(val_level)
            node_levels.append(node_level)
            level += 1
            val_level, node_level = [], []