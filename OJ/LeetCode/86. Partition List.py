# 使用快慢两个指针，一个保存小于的，另一个保存大于等于的结点
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        node1 = less = ListNode(0)
        node2 = bigger = ListNode(0)

        while head:
            if head.val < x:
                less.next = head
                less =  less.next
            else:
                bigger.next = head
                bigger = bigger.next
            head = head.next


        bigger.next = None
        less.next = node2.next

        return node1.next
        