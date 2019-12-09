# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if pre == tin == []:
            return
        root = TreeNode(pre[0])
        newltin = tin[:tin.index(pre[0])]
        newrtin = tin[tin.index(pre[0]) + 1:]
        newlpre = [i for i in pre if i in newltin]
        newrpre = [i for i in pre if i in newrtin]
        root.left = self.reConstructBinaryTree(newlpre, newltin)
        root.right = self.reConstructBinaryTree(newrpre, newrtin)
        return root

    def reConstructBinaryTree2(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        root = TreeNode(pre[0])
        if len(pre) == 1:
            return root
        left_len = tin.index(pre[0])
        right_len = len(tin) - left_len - 1
        root.left = self.reConstructBinaryTree2(pre[1:left_len + 1], tin[:left_len])
        # right_len 为0时，可能出错
        root.right = self.reConstructBinaryTree2(pre[len(tin)-right_len:], tin[len(tin)-right_len:])
        return root

    def inOrder(self, node):
        if node is None:
            return
        self.inOrder(node.left)
        print(node.val)
        self.inOrder(node.right)


if __name__ == '__main__':
    pre, tin = [1,2,4,3,5,6],[4,2,1,5,3,6]

    s = Solution()
    ret = s.reConstructBinaryTree2(pre, tin)
    s.inOrder(ret)