# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    #思路1：遍历链表，保存所有节点信息，然后在保存信息中处理，重新构建删除后的链表
    def removeNthFromEnd(self, head, n):
        indexs = []
        cur = head
        while cur:
            indexs.append(cur)
            cur = cur.next
        target_index_inorder = len(indexs) - n
        # 第一位没有上一位
        if n == len(indexs):
            print(0)
            return head.next
        # 倒数第一位没有下一位
        elif n == 1:
            print(1)
            indexs[- (n+1)].next = None
        else:
            print(2)
            indexs[- (n+1)].next = indexs[- (n-1)]
        return head
    

    # 思路2:快慢指针
    def removeNthFromEnd_2(self, head: ListNode, n: int) -> ListNode:
        start = ListNode()
        fast, slow = start, start
        start.next = head
        for _ in range(n + 1):
            fast = fast.next
        while fast is not None:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return start.next
