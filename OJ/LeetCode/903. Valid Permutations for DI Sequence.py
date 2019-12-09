class Solution:
    def numPermsDISequence(self, S):
        dp = [1] * (len(S) + 1)  # S比N少一位
        for c in S:
            if c == "I":
                # 下一个更大
                dp = dp[:-1]  # 最后一个数不可能有比他大
                cur = 0
                for i in range(len(dp)):
                    # 每一个i都是0-i之和
                    cur += dp[i]
                    dp[i] = cur
            else:
                # 下一个更小
                dp = dp[1:] # 第一个不可能有比他小
                cur = 0
                for i in range(len(dp)-1, -1, -1):
                    # 每一个
                    cur += dp[i]
                    dp[i] = cur
        return dp[0] % (10**9 + 7)


    def numPermsDISequence2(self, S):
        n = len(S)
        dp = [[0] * (n + 1)] * (n + 1)
        
        for i in range(n + 1):
            dp[0][j] = 1

        for i in range(n):
            if S[i] == "I":
                pass


if __name__ == "__main__":
    print(Solution().numPermsDISequence1("IDD"))