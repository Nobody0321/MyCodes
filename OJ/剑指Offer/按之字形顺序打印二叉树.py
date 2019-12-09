# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 因为是之字形，所以需要2个数组，分别存储相邻的两行，
    # 来回倒，实现从两个方向读取
    def Print(self, pRoot):
        flag = 1
        stack1 = []  # 存放奇数层
        stack2 = []  # 存放偶数层
        if pRoot == None:
            return None
        stack1.append(pRoot)
        ret = []
        while stack1!=[] or stack2!=[]:
            if flag > 0:
                # 说明当前要处理奇数层
                tmp = []
                while len(stack1):
                    node = stack1.pop()
                    if node != None:
                        tmp.append(node.val)
                        print(node.val,end = ' ')
                        stack2.extend([node.left, node.right])

                if tmp!=[]:
                    ret.append(tmp)
                    flag *= -1
                    print()
            else:
                # 处理偶数层
                tmp = []
                while len(stack2):
                    node = stack2.pop()
                    if node != None:
                        tmp.append(node.val)
                        print(node.val,end = ' ')
                        stack1.extend([node.right, node.left])

                if tmp!=[]:
                    ret.append(tmp)
                    flag *= -1
                    print()
        return ret