# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
   def mergeTwoLists(self, l1, l2):
        res = ListNode(0)
        if l1 == None:
            return l2
        if l2 == None:
            return l1

        if l1.val < l2.val:
            res = l1
            res.next = self.mergeTwoLists(l1.next,l2)
        else:
            res = l2
            res.next = self.mergeTwoLists(l2.next,l1)

        return res

if __name__ == "__main__":
    s = Solution()
    print(s.mergeTwoLists())