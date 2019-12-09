class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        Inorder(pRootOfTree)
        
    def Inorder(self, node):
        Inorder(node.left)
        s.append(node.val)
        Inorder(node.right)