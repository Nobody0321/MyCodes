# 剑指offer出现过
# 矩阵每一行都是左到右递增，每一列都是上到下递增，所以从左下角开始，类似二分查找
class Solution:
    def searchMatrix(self, matrix, target):
        if matrix == []:
            return False
        height, width = len(matrix), len(matrix[0])
        i, j= height-1, 0
        while 0<=i<=height-1 and 0<=j<=width-1:
            if 0<=i<=height-1 and 0<=j<=width-1 and matrix[i][j] < target:
                j+=1
            elif 0<=i<=height-1 and 0<=j<=width-1 and matrix[i][j] > target:
                i-=1
            else:
                return True
        return False

if __name__ == "__main__":
    print(Solution().searchMatrix([[1]], 2))