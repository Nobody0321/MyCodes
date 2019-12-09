class Solution:
    def FindContinuousSequence(self, S):
        # write code here
        # 使用双指针/滑动窗口，两个指针确定一个窗口
        result = []
        i, j = 1, 2
        while j > i:
            total = (i+j)*(j-i+1)/2
            if total == S:
                r = []
                for k in range(i,j+1):
                    r.append(k)
                result.append(r)
                i += 1
            elif total < S:
                j += 1
            else:
                i += 1
        return result