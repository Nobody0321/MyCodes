# 可以记住，n&(n-1)就是统计二进制中1的个数
class Solution:
    def NumberOf1(self, n):
        # write code here
        # 由于python没有位数限制，我们需要手动把负数转成32位的补码，否则由于无限位数，程序会一直循环
        cnt= 0
        n = n if n >= 0 else n & 0xffffffff
        while n != 0:
            cnt += 1
            n = n & (n-1)
        return cnt