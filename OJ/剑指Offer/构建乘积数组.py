class Solution:
    # # 解法1 O(N^2)
    # def multiply(self, A):
    #     return [self.mul(i,A) for i in range(len(A)) ]

    # def mul(self,i,A):
    #     s = 1
    #     for idx, num in enumerate(A):
    #         if idx == i:
    #             continue
    #         s *= num
    #     return s

    # 解法2 每一个B[i]都是A[0]*...*A[i-1]  *  A[i+1] *...*A[n-1]
    # 分别计算这两部分就可以
    def multiply(self, A):
        # 计算前半部分
        b = [1] # B[0] 没有前半部分（为1）
        for i in range(len(A)-1):
            b.append(b[-1]*A[i])
        
        # 计算后半部分
        t = 1
        for i in reversed(range(len(A))):
            b[i] = b[i] * t # B[n-1] 没有后项（后项为1）
            t = t * A[i]    # 开始累计后半部分
        return b


if __name__ == "__main__":
    print(Solution().multiply([1,2,3,4,5]))