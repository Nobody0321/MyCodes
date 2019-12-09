# leetcode 72. edit distance
class Solution:
    def minDistance(self, word1, word2):
        # 递归
        # 1. 末位相同，d[m][n] = d[m-1][n-1]
        # 2. 末位不同
        #     2.1 因为可以删除
        #       d[m][n] = d[m-1][n] + 1
        #     2.2 因为可以添加，添加最后一位使相同
        #         由1：添加后:
        #             d[m][n] = d[m+1-1][n-1] + 1
        #         =>  d[m][n] = d[m][n-1] + 1
        #     2.3 因为可以替换：
        #         d[m][n] = d[m-1][n-1] + 1
        m, n = len(word1), len(word2)
        if m == 0:
            return n
        elif n == 0:
            return m
        else:
            m, n = m-1, n-1
            if word1[m] == word2[n]:
                return self.minDistance(word1[:m], word2[:n])
            else:
                r1 = self.minDistance(word1[:m], word2) + 1
                r2 = self.minDistance(word1, word2[:n]) + 1
                r3 = self.minDistance(word1[:m], word2[:n]) + 1
                return min(r1,r2,r3)

        
    def minDistance_2(self, word1, word2):
    # 2. 动态规划，因为递归超时了 dp 范围是0：0~l1:l2
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

if __name__ == "__main__":
    print(Solution().minDistance_2("spartan","part"))