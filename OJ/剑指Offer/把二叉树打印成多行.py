# 剑指offer
# 思路：使用一个变量记录当前节点的层数
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def __init__(self):
        self.levels = []
          
    def Print(self, pRoot):
        # write code here
        def depth(node, d):
            if node == None:
                return 
            if d > len(self.levels):
                self.levels.append([])            
            self.levels[d-1].append(node.val)
            depth(node.left, d+1)
            depth(node.right, d+1)
        depth(pRoot,1)
        return self.levels