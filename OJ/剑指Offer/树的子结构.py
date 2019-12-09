class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        result = False
        if pRoot1 and pRoot2: 
            if(pRoot1.val == pRoot2.val):
                result =  self.IsSameTree(pRoot1, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)
        return result   

    def IsSameTree(self, node1, node2):
        if node2 == None:
            return True
        elif node1 == None:
            return False
        elif node1.val != node2.val:
            return False
        else:
            return self.IsSameTree(node1.left, node2.left) and self.IsSameTree(node1.right, node2.right)