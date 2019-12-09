# 思路，类似unique path 1 但是遇到路障要将路径数置零
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        if obstacleGrid == []:
            return None
        height = len(obstacleGrid)
        width = len(obstacleGrid[0])
        dp = [[0] * width for _ in range(height)]

        for i in range(height):
            if obstacleGrid[i][0]:
                break
            else:
                dp[i][0] = 1

        for i in range(width):
            if obstacleGrid[0][i]:
                break
            else:
                dp[0][i] = 1

        for i in range(1,height):
            for j in range(1,width):
                if obstacleGrid[i][j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[height-1][width-1]
            

if __name__ == "__main__":
    grid = [[0,0,0], [0,1,0], [0,0,0]]
    print(Solution().uniquePathsWithObstacles(grid))