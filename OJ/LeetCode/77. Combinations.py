class Solution:
    def combine(self, n, k):
        self.ret = []
        self.do([], 1, n ,k)
        return self.ret
    
    def do(self, tmp, start, n, k):
        if k == 0:
            # 说明路径长度到头
            self.ret.append(tmp[:])
            # 之前用self.ret.append(tmp)，会传引用进来，最后导致结果全空
        else:
            # 否则路径还可以继续
            for i in range(start, n+1):
                tmp.append(i)
                self.do(tmp, i+1, n, k-1)
                # 执行了一次循环后tmp会加上一个当前的i，这里需要pop掉，以便下一次循环
                tmp.pop()

if __name__ == "__main__":
    print(Solution().combine(n=4,k=2))