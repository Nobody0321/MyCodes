class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
# class Solution:
#     # 返回 RandomListNode
#     def Clone(self, pHead):
#         import copy
#         return copy.deepcopy(pHead)

class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if not pHead:
           return None
        newHead = RandomListNode(pHead.label)
        newHead.random = pHead.random
        newHead.next = self.Clone(pHead.next)
        return newHead