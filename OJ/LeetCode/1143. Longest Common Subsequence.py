class Solution:
    def longestCommonSubsequence(self, text1, text2):
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                dp[i][j] = dp[i-1][j-1] + 1  if text1[i - 1] == text2[j - 1] else max(dp[i][j-1], dp[i-1][j])
        return dp[-1][-1]
    

if __name__ == "__main__":
    print(Solution().longestCommonSubsequence("ezupkr", "ubmrapg"))