# 剑指offer
# 因为是有序链表，所以用一个指针去找到下一个不重复数字，然后接上
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead == None or pHead!=None and pHead.next == None:
            return pHead
        
        current = ListNode(None)            
        
        if(pHead.next.val == pHead.val):
            current = pHead.next.next
            while current!=None and current.val == pHead.val:
                current = current.next
            return self.deleteDuplication(current)

        else:
            current = pHead.next
            pHead.next = self.deleteDuplication(current)
            return pHead