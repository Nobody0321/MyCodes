# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 1. 使用快慢指针找到倒数第k个结点
    # 2. 从第k个结点开始倒转
    def rotateRight(self, head, k):
        if not head:
            return None
        slow = fast = head

        l, h = 0, head
        while h:
            h = h.next
            l += 1
        k = k % l

        if k == 0:
            return head
            
        while k != 0:
            fast = fast.next
            k -= 1

        while fast.next:
            slow, fast = slow.next, fast.next
            # slow指向了倒数第k+1
        ret = slow.next
        slow.next = None
        fast.next = head
        return ret