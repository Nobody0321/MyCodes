# 剑指offer
# 使用类似桶排序的思想
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        d = [False]*len(numbers) # boolean 比 int 短
        for i in numbers:
            if d[i] == True:
                duplication[0] = i
                return True
            else:
                d[i] = True
        return False