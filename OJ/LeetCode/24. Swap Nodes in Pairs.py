# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def swapPairs_0(self, head: ListNode) -> ListNode:
        """ 用两个指针，一次替换两个

        Args:
            head (ListNode): _description_

        Returns:
            ListNode: _description_
        """
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return self.next