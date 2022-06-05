# 一看就是用dp吧
# dp[i][j] 表示是s[i ~ j] 的回文与否
# dp[i][i] = 1
# df[i][j]
class Solution:
    def longestPalindrome(self, s):
        dp = [[0] * len(s) for _ in range(len(s))]

        max_result = [0, [0, 0]]
        for i in range(len(s)):
            dp[i][i] = True
            for j in range(i + 1, len(s)):
                dp[i][j] = (dp[i + 1][j - 1] or (j - i <= 1)) and s[i] == s[j]
                if dp[i][j] and j - i + 1 > max_result[0]:
                        max_result[0] = j - i + 1
                        max_result[1] = [i, j]

        return s[max_result[1][0]: max_result[1][1]+1]


if __name__ == "__main__":
    # s = "babad"
    # print(Solution().longestPalindrome(s))

    # s = "cbbd"
    # print(Solution().longestPalindrome(s))

    s = "aaaa"
    print(Solution().longestPalindrome(s))
