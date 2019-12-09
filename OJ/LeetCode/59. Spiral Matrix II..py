class Solution:
    def generateMatrix(self, n):
        matrix =[[0]*n for _ in range(n)]
        maxValue = n*n+1
        minRow, maxRow, minCol, maxCol = 0, n-1, 0, n - 1
        value = 1
        while value != maxValue:
            for i in range(minCol,maxCol+1):
                matrix[minRow][i] = value
                value += 1
            minRow += 1  # 填充了第一行
            for i in range(minRow, maxRow+1):
                matrix[i][maxCol] = value
                value += 1
            maxCol -= 1  # 填充了最后一列
            for i in reversed(range(minCol, maxCol+1)):
                matrix[maxRow][i] = value
                value += 1
            maxRow -= 1  # 填充了最后一行
            for i in reversed(range(minRow, maxRow+1)):
                matrix[i][minCol] = value
                value +=1
            minCol += 1   
        return matrix


if __name__ == "__main__":
    print(Solution().generateMatrix(3))