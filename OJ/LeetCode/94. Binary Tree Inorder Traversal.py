# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def inorderTraversal(self, root):
        result = []

        def dfs(node, res):
            if node:
                dfs(node.left, res)
                res.append(node.val)
                dfs(node.right, res)

        dfs(root, result)
        return result

    def inorderTraversal2(self, root: TreeNode) -> List[int]:
        s = []
        ret = []
        while root or s != []:
            while root:
                # find the first left leaf
                s.append(root)
                root = root.left
            # root is None now
            if len(s):
                root = s.pop()
                ret.append(root.val)
                root = root.right
        return ret
