class Solution:
    # matrix类型为二维列表，需要返回列表
    def __init__(self):
        self.ret = []
    def printMatrix(self, matrix):
        self.draw(matrix, 0, 0, len(matrix[0]), len(matrix))
        return self.ret
    
    def draw(self,matrix,x,y,width,height):
        if  (width<=0 or height<=0):
            return 
        new_x, new_y = x + 1, y + 1
        if height == 1:
            for i in range(width):
                self.ret.append(matrix[x][y+i])
            return
        elif width == 1:
            for i in range(height):
                self.ret.append(matrix[x+i][y])
            return
        else:
            for i in range(0,width):
                self.ret.append(matrix[x][y+i])
            for i in range(1, height):
                self.ret.append(matrix[x+i][y+width-1]) 
            for i in range(1,width):
                self.ret.append(matrix[x+height-1][y+width-1-i])
            for i in range(1,height-1):
                self.ret.append(matrix[x+height-1-i][y])
            self.draw(matrix, new_x, new_y, width -2, height -2)

if __name__ == "__main__":
    # S= [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    S=[[1],[2],[3],[4],[5]]
    s = Solution()
    print(s.printMatrix(S))
    # print(s.ret)