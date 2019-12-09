# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def sortedListToBST(self, head):
        array = []
        while head:
            array.append(head.val)
            head = head.next
        return self.buildTree(array)

    def buildTree(self, array):
        l = len(array)
        if l == 0:
            return
        rootIdx = array.index(array[l>>1])
        root = TreeNode(array[rootIdx])
        root.left = self.buildTree(array[:rootIdx])
        root.right = self.buildTree(array[rootIdx+1:])
        return root