# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def buildTree(self, preorder, inorder):
        # 思路：前序遍历第一个必是根节点，因此中序可以分为左右子树，
        # 知道左右子树长度后自然知道了左右子树前序的范围，然后递归左右子树
        rootIdx_inorder = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[rootIdx_inorder])
        root.left = self.buildTree(preorder, inorder[0:rootIdx_inorder])
        root.right = self.buildTree(preorder, inorder[rootIdx_inorder+1:])
        return root


if __name__ == "__main__":
    print(Solution().buildTree(preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]))