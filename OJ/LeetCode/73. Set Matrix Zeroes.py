class Solution:
    def setZeroes(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 1.笨办法，找出所有有零的行和列
        width = len(matrix[0])
        height = len(matrix)
        rows = []
        cols = []
        for i in range(height):
            for j in range(width):
                if matrix[i][j] == 0:
                    if i not in rows:
                        rows.append(i)
                    if j not in cols:
                        cols.append(j)

        for row in rows:
            for i in range(width):
                matrix[row][i] = 0
        
        for col in cols:
            for i in range(height):
                matrix[i][col] = 0
