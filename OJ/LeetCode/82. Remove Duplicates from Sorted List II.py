# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 使用两个指针，
# 还可以使用递归，查找下一个非重复结点并返回
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 1. 如果是长度为0或1，直接返回
        if not head or head and not head.next:
            return head
        
        # 2. current 跳过重复结点
        cur = ListNode(None)
        if head.next.val == head.val:
            cur = head.next.next
            while cur and cur.val == head.val:
                # 持续跳过重复节点
                cur = cur.next
            # 返回第一个非重复结点
            return self.deleteDuplicates(cur)
        else:
            # head.next就是非重复结点
            cur = head.next
            head.next = self.deleteDuplicates(cur)
            return head