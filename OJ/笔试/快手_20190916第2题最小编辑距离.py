def minDistance(self, word1, word2):
    l1, l2 = len(word1), len(word2)
    dp = [[0 for _ in range(l2+1)] for _ in range(l1+1)]
    dp[0][0] = 0
    for i in range(l1+1):
        for j in range(l2+1):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                dp[i][j] = dp[i][j-1] + 1
            elif j == 0:
                dp[i][j] = dp[i-1][j] + 1
            else:
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1

    return dp[l1][l2]

str1, str2 = input(), input()
print(minDistance(str1, str2))