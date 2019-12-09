class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def __init__(self):
        self.result = []

    def FindPath(self, root, target):
        self.dfs(root, target, [])
        return self.result

    def dfs(self, node, target, path):
        if node == None:
            return self.result
        curpath = path +[node.val]
        target -= node.val
        if target == 0 and node.left == None and node.right == None:
            self.result.append(curpath)
        elif target > 0:
            self.dfs(node.left, target, curpath) 
            self.dfs(node.right, target, curpath)


if __name__ == "__main__":
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right=TreeNode(12)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(7)
    print(Solution().FindPath(root,22))
