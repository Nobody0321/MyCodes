class Solution:
    def numDecodings(self, s):
        # 动态规划,对于226,首先看2,22：
        # 2 只有2 一个解
        # 22可以拆成2,2或者22,
        # 226可以分成2,2,6和22,6 或者 2，26
        # 即22和6（2,2,6/22,6）或者2和26（2,26）
        # dp[226] = dp[2] + dp[22]    以26为或者6为结尾，有两种情况
        # 227 可以分成2,2,7或22,7
        # dp[227] =dp[22]
        # 为了维持dp需要有一个初始值
        l = len(s)
        if l == 1 and s == '0' or s[0] == '0':
            return 0
        elif l == 1:
            return 1
        dp = [0] * (l + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(1, l):
            s1 = ord(s[i])-ord('0')
            s2 = ord(s[i-1])-ord('0')
            if s1 == 0 and s2 == 0 or s1 == 0 and s2 * 10 + s1 > 26:
                # 例如80或者00 这样的,没法计算
                return 0
            elif s2 == 0  or  s2 * 10 + s1 > 26:
                # 加了一个不可分解数字比如dp[227]= dp[22]或者dp[207]=dp[20]
                dp[i+1] += dp[i]
            elif s1 == 0:
                # 当前数字为0, 220,导致不能分成22,0,退化成了dp[220] = dp[2]
                dp[i+1] += dp[i-1] 
            else:
                dp[i+1] += dp[i] + dp[i-1]
        return dp[-1]


if __name__ == "__main__":
    print(Solution().numDecodings('01'))