# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reverseBetween(self, head, m, n):
        h = ListNode(0)
        h.next = head
        pre = h
        for i in range(m-1):
            pre = pre.next
        # pre 现在指向逆序部分之前的节点
        start = pre.next
        then = start.next