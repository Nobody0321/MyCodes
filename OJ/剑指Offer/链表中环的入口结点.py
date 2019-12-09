# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 解法一，路过一个节点就打断它的next指针，这样最后就会停止在环的入口结点
    def EntryNodeOfLoop(self, pHead):
        if pHead.next:
            a, faster = pHead, pHead.next
            while faster:
                a.next = None
                a = faster
                faster= faster.next
            return a
        return None