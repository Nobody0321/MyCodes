# 思路： 因为机器人只能向右或者向下移动，
# 所以一个点上的机器人必然是从上或者从左来的
# 即 dp[i][j] = dp[i-1][j] + dp[i][j-1]
class Solution:
    def uniquePaths(self, m, n):
        dp = [[0]*n]*m
        # 1.处理边上：从原点出发到相接两边任意一点，都只有一条路
        for i in range(m):
            dp[i][0] = 1
        for i in range(n):
            dp[0][i] = 1
        
        # 2. 处理其他部分
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
        return dp[m-1][n-1]



if __name__ == "__main__":
    print(Solution().uniquePaths(3,2))