# 剑指offer 二叉树的下一个节点
# # class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
# 0.空节点
# 1.如果有右子树,结果是右子树的最左儿子
# 2.如果没有右子树,结果是把当前节点所在部分当作左子树的第一个节点
class Solution:
    def GetNext(self, pNode):
        if not pNode:
            return pNode
        elif pNode.right:
            pNode = pNode.right
            while pNode.left:
                pNode = pNode.left
            return pNode
        else:
            while pNode.next:
                parent = pNode.next
                if parent.left == pNode:
                    return parent
                else:
                    # 向上遍历
                    pNode = pNode.next
            return None