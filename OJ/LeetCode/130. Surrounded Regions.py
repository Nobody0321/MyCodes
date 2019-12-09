class Solution:
    def solve(self, board):
        """
        Do not return anything, modify board in-place instead.
        """
        if board == []:
            return
        height = len(board)
        width = len(board[0])
        def dfs(board,x,y):
            if x < 0 or x >= height or y < 0 or y >= width or board[x][y] != 'O':
                return
            board[x][y] = '#'
            dfs(board, x-1, y)
            dfs(board, x, y-1)
            dfs(board, x+1, y)
            dfs(board, x, y+1)

            
        for i in range(height):
            dfs(board, i, 0)
            dfs(board, i, width-1)
            
        for j in range(width):
            dfs(board, height-1, j)
            dfs(board, 0, j)
            
        for i in range(height):
            for j in range(width):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                    
        for i in range(height):
            for j in range(width):
                if board[i][j] == '#':
                    board[i][j] = 'O'
        
if __name__ == "__main__":
    board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    Solution().solve(board)
    print(board)