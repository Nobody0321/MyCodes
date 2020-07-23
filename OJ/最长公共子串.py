class Solution:
    def longestCommonSubsequence(self, text1, text2):
    # 二维DP，先从两个str的第一位开始，与另一串任何一位相同，就是一个长度为1 的公共子串
        max_len = 0
        dp = [[0] * len(text2) for _ in range(len(text1))]
        for i in range(len(text1)):
            dp[i][0] = int(text1[i] == text2[0])
        for i in range(len(text2)):
            dp[0][i] = int(text1[0] == text2[i])
        for i in range(1, len(text1)):
            for j in range(1, len(text2)):
                dp[i][j] = dp[i - 1][j - 1] + 1 if text1[i] == text2[j] else 0
                max_len = max(max_len, dp[i][j])
        return max_len

    def longestCommonSubsequence2(self, text1, text2):
        # 二维dp每次都是与左上方的值有关，那么就一维地求所有左上到右下直线
        l, max_l = 0, 0
        # 从右上到左下，斜45°扫描dp矩阵
        row, col = 0, len(text2) - 1
        while row < len(text1):
            i, j = row, col
            while i < len(text1) and j < len(text2):
                if text1[i] == text2[j]:
                    l += 1
                    max_l = max(l, max_l)
                else:
                    l = 0
                i += 1
                j += 1
            if col > 0:
                col -= 1
            else:
                row += 1
        return max_l
        


if __name__ == "__main__":
    str1="1AB2345CD"
    str2="12345EF"
    print(Solution().longestCommonSubsequence2(str1, str2))