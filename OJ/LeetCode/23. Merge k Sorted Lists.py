# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeKLists(self, lists):
        s = []
        for each in lists:
            while each:
                s.append(each.val)
                each = each.next
        s.sort()
        head = ListNode(0)
        h = head
        for i in s:
            head.next = ListNode(i)
            head = head.next
        return h