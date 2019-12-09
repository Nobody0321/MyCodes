# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    #思路1：遍历链表，保存所有节点信息，然后在保存信息中处理，重新构建删除后的链表
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        result = []
        ret = ListNode(0)
        retHead = ret
        while(head):
            result.append(head.val)
            head = head.next
        print(result)
        length = len(result)
        for i in range(length):
            if i == length-n:
                continue
            ret.next = ListNode(0)
            ret = ret.next
            ret.val = result[i]
        return retHead.next

    # 思路2:想办法n的正序,直接o(n)遍历就行
    # 
    def removeNthFromEnd_2(self, head: ListNode, n: int) -> ListNode:
        start = ListNode(0)
        fast, slow = start, start
        slow.next = head
        for i in range(n):
            fast.next = ListNode(0)
            fast = fast.next
        while fa
