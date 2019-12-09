class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    # 返回合并后列表
    def Merge1(self, pHead1, pHead2):
        # 非递归
        head = ListNode(0)
        ret = head
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                head.next = ListNode(pHead1.val)
                head = head.next
                pHead1 = pHead1.next
            else:
                head.next = ListNode(pHead2.val)
                head = head.next
                pHead2 = pHead2.next
        if pHead1:
                head.next = pHead1
        else:
            head.next = pHead2

        return ret.next

    def Merge2(self, pHead1, pHead2):
        # 递归，看起来像是把两个链表串起来
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        if pHead1.val <= pHead2.val:
            pHead1.next = self.Merge2(pHead1.next, pHead2)
            return pHead1
        else:
            pHead2.next = self.Merge2(pHead2.next, pHead1)
            return pHead2
        
