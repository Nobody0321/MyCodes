
class Solution:
    def minPathSum(self, grid):
        # 贪心
        # 这个方法超时了
        if grid == []:
            return None
        height, width = len(grid), len(grid[0])

        def findPath(i,j):
            if (not 0<= i < height) or not ( 0<=j <width):
                return float('inf')
            elif (i,j) == (height-1,width-1):
                return grid[i][j]
            else:
                return min(findPath(i+1,j), findPath(i, j+1)) + grid[i][j]
            
        return findPath(0,0)

    def minPathSum_2(self, grid):
        # 类似动态规划,使用一个数组记录对应点的当前路径数\
        if grid == []:
            return None
        height, width = len(grid), len(grid[0])
        pathSum = [[0]*width]*height

        for i in range(height):
            for j in range(width):
                if i ==0 and j == 0:
                    # 处理原点
                    pathSum[i][j] = grid[0][0]
                # 下面处理原点所在的两条边，规则决定这两条边上的点只可能是从同边上的上一个点走过来的
                if i == 0 and j > 0:
                    pathSum[i][j] = pathSum[i][j-1] + grid[i][j]
                if j == 0 and i > 0:
                    pathSum[i][j] = pathSum[i-1][j] + grid[i][j]
                else:
                    # 处理剩下的点，这些点可能是从上方或者左边走过来的
                    pathSum[i][j] = min(pathSum[i][j-1], pathSum[i-1][j]) + grid[i][j]
        return pathSum[-1][-1]


if __name__ == "__main__":
    print(Solution().minPathSum([[1,3,1],[1,5,1],[4,2,1]]))