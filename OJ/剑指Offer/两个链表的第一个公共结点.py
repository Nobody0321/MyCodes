class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 解法1，公共结点到两个链表尾部的距离是一样的，所以让较长的链表先走k步，使得两个链表同步
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        l1 = self.findLength(pHead1)
        l2 = self.findLength(pHead2)
        while l1 > l2:
            pHead1 = pHead1.next
            l1 -= 1
        while l2 > l1:
            pHead2 = pHead2.next
            l2 -= 1

        while pHead1 != None:
            if pHead1 == pHead2:
                return pHead1
            pHead1 = pHead1.next
            pHead2 = pHead2.next  
        return None

    def findLength(self, head):
        l = 0
        while head != None:
            l += 1
            head = head.next
        return l