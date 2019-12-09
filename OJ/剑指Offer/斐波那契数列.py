# 快速求幂算法
# 例如
# 3 ^ 999 = 3 ^ (512 + 256 + 128 + 64 + 32 + 4 + 2 + 1)
#         = (3 ^ 512) * (3 ^ 256) * (3 ^ 128) * (3 ^ 64) * (3 ^ 32) * (3 ^ 4) * (3 ^ 2) * 3
# 999的二进制是1111100111 正好对应了3的指数部分
# 又因为可以通过辗转相除法依次求出一个十进制数从低到高的所有二进制（6可以得到011）
class Solution:
    def Fibonacci(self, n):
        # write code here
        ret = [0,1]
        if n <= 1:
            return ret[n]
        for i in range(2,n+1):
            ret.append(ret[i-1]+ret[i-2])
        return ret[-1]
        

if __name__ == "__main__":
    print(Solution().Fibonacci(5))