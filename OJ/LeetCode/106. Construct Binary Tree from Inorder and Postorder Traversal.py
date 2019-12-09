# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def buildTree(self, inorder, postorder):
        # 思路：后序遍历第一个必是根节点，因此中序可以分为左右子树，
        # 知道左右子树长度后自然知道了左右子树后序的范围，然后递归左右子树
        if inorder == []:
            return None
        if len(inorder) == 1 and len(postorder) == 1:
            return TreeNode(inorder[0])
        rootVal = postorder[-1]
        root = TreeNode(rootVal)
        rootIdx_inorder = inorder.index(rootVal)
        leftTree_inorder, rightTree_inorder = inorder[:rootIdx_inorder], inorder[rootIdx_inorder+1:]
        leftTree_postorder, rightTree_postorder = postorder[:len(leftTree_inorder)], postorder[len(leftTree_inorder):-1]
        root.left = self.buildTree(leftTree_inorder, leftTree_postorder)
        root.right = self.buildTree(rightTree_inorder, rightTree_postorder)
        return root


if __name__ == "__main__":
    root = Solution().buildTree(inorder = [9,3,15,20,7], postorder = [9,15,7,20,3])
    def inorder(node):
        if node:
            if node.left:
                inorder(node.left)
            print(node.val)
            if node.right:
                inorder(node.right)
        else:
            return 

    def postorder(node):
        if node:
            if node.left:
                inorder(node.left)
            if node.right:
                inorder(node.right)
            print(node.val)
        else:
            return 

    # inorder(root)
    postorder(root)