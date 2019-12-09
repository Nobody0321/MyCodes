# 使用一个矩阵记录当前节点是否已访问，然后不断回溯，注意边界条件即可。与小米面试的n sum 类似
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        i = 0
        m = []
        for _ in range(rows):
            m.append(matrix[i:i+cols])
            i += cols
        matrix = m

        self.visited = [[0] * cols for _ in range(rows)]
        for x in range(rows):
            for y in range(cols):
                if self.findnext(matrix,x,y,rows,cols,path,0):
                    return True
                
    
    def findnext(self, matrix, x, y, rows, cols, path, idx):
        if x<0 or x>=rows or y<0 or y>=cols or self.visited[x][y]== 1 or matrix[x][y] != path[idx]:
            return False
        
        if idx ==len(path)-1:
            return True
        
        self.visited[x][y] = 1

        if self.findnext(matrix,x+1,y,rows,cols,path,idx+1) or \
            self.findnext(matrix, x-1, y, rows, cols, path, idx+1) or \
            self.findnext(matrix, x, y+1, rows, cols, path, idx+1) or \
            self.findnext(matrix, x, y-1, rows, cols, path, idx+1):
            return True
        else:
            self.visited[x][y] = False
        return False


if __name__ == "__main__":
    print(Solution().hasPath(['a','b','c','e','s','f','c','s','a','d','e','e'],3,4,'bcced'))