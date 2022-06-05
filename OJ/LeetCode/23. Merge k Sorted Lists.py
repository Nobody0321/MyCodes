# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def merge_2_lists(self, l1, l2):
        """
        return merged list from 2 list
        """
        res = ListNode()
        cur = res
        if l1 is None and l2 is None:
            return
        elif l1 is None:
            return l2
        elif l2 is None:
            return l1

        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                cur.next = l1
                cur = cur.next
                l1 = l1.next
            elif l1.val >= l2.val:
                cur.next = l2
                cur = cur.next
                l2 = l2.next
        if l1 is not None:
            cur.next = l1
        elif l2 is not None:
            cur.next = l2
        return res.next

    def merge_k_lists_dividing(self, i, j, lists):
        if i == j:
            return lists[i]
        else:
            mid = int((i + j)  // 2)
            left_result = self.merge_k_lists_dividing(i, mid, lists)
            right_result = self.merge_k_lists_dividing(mid + 1, j, lists)
            return self.merge_2_lists(left_result, right_result)
        

    def mergeKLists(self, lists):
        """
        使用分治的思想，对每两个list merge
        """
        if len(lists) == 0:
            return
        else:
            return self.merge_k_lists_dividing(0, len(lists)-1, lists)
