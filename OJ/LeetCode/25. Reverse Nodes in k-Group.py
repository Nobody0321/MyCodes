# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 按顺序把所有节点存到list，最后逆转部分
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        nodes = []
        while head:
            nodes.append(head)
            head= head.next
        nodes = list(reversed(nodes[:k])) + nodes[k:]
        h = ListNode(0)
        rhead = h
        for each in nodes:
            h.next = each
            h = h.next
        return rhead.next