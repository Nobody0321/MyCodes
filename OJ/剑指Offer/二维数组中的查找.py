# 思路：
# 对于左下角的元素，向右递增，向上递减
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        width = len(array[0])
        height = len(array)
        i, j = height-1, 0
        while i>=0 and j < width:
            if array[i][j] == target:
                return True
            if array[i][j] > target:
                i -= 1
            elif array[i][j] < target:
                j += 1
        return False