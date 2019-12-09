# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if pHead is None:
            return
        pre = None
        n = None

        while pHead:
            n = pHead.next
            pHead.next = pre

            pre = pHead
            pHead = n
        return pre
