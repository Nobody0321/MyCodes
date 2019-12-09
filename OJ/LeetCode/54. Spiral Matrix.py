class Solution:
    def spiralOrder(self, matrix):
        if len(matrix) == 0:
            return 
        width = len(matrix[0])
        height = len(matrix)
        result = []
        self.get(matrix, width, height, result)
        return result[:width*height]

    def get(self, matrix , width, height, result):
        if width < 0 or height < 0:
            return
            
        x = (len(matrix[0])-width) // 2
        y = (len(matrix)-height) // 2


        for i in range(width):
            result.append(matrix[x][y+i])

        for i in range(1, height):
            result.append(matrix[x + i][y + width - 1])

        for i in range(1, width):
            result.append(matrix[x+height-1][y+width-1-i])

        for i in range(1,height-1):
            result.append(matrix[x+height-1-i][y])

        self.get(matrix,width-2,height-2,result)

if __name__ =='__main__':
    matrix = [[1,2,3,4,5],[5,6,7,8,9],[9,10,11,12,13]]
    result = Solution().spiralOrder(matrix)
    print(result)