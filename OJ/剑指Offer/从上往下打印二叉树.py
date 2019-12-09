# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def __init__(self):
        import Queue
        self.level = Queue.Queue()
        self.nodes = []

    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return self.nodes
        self.level.put(root)
        while not self.level.empty():
            curNode = self.level.get()
            if curNode.left:
                self.level.put(curNode.left)
            if curNode.right:
                self.level.put(curNode.right)
            self.nodes.append(curNode.val)
        return self.nodes