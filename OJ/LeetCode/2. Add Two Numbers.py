# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        carry = 0
        n1, n2 = l1, l2
        r = ret = ListNode(0)
        while n1 and n2:
            # 从低位开始相加，顺便进位
            ret.next = ListNode(0)
            ret = ret.next
            cur = carry + n1.val + n2.val
            ret.val = cur % 10
            carry = cur // 10
            n1, n2 = n1.next, n2.next          
        while n1:
            ret.next = ListNode(0)
            ret = ret.next
            cur = carry + n1.val
            ret.val = cur % 10
            carry = cur // 10
            n1 = n1.next
        while n2:
            ret.next = ListNode(0)
            ret = ret.next
            cur = carry + n2.val
            ret.val = cur % 10
            carry = cur // 10
            n2 = n2.next
        if carry:
            ret.next = ListNode(0)
            ret = ret.next
            ret.val = carry
        return r.next
