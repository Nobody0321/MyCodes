# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # 中序遍历后返回对应的节点，后续尝试一下非递归版本的中序遍历
        tree = []
        def dfs(node):
            if node == None:
                return 
            dfs(node.left)
            tree.append(node)
            dfs(node.right)
        dfs(pRoot)
        if k <=0 or k> len(tree):
            return None
        return tree[k-1]