# 1. 非递归版：
# 使用三个指针pre,cur,tmp三个指针
# pre cur 主要负责翻转两个节点的指向，tmp用来记录原本顺序的下一个节点

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def reverseList(head):
    if head == None or head.next == None:
        return head
    
    pre, cur, tmp = head, head.next, head.next.next
    while cur:
        tmp = cur.next
        cur.next = pre  # 反转cur pre这两个节点
        pre = cur
        cur = tmp
    
    head.next = None
    return pre