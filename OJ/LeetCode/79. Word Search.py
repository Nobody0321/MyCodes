class Solution:
    # 也是一个地图路径问题，尝试使用回溯/递归/dfs的方法解决
    def exist(self, board, word):
        self.board = board
        self.height, self.width = len(board), len(board[0])
        
        for i in range(self.height):
            for j in range(self.width):
                if  self.dfs(i,j,word):
                    return True
        return False

    def dfs(self,i, j, word):
        if self.board[i][j] == word[0]:
            # 掩盖当前点的字符
            self.board[i][j] = '#'
            # 递归查找
            if len(word) == 1 or (i > 0 and self.dfs(i-1, j,word[1:])) or (i < self.height-1 and self.dfs(i+1, j, word[1:])) or (j > 0 and self.dfs(i, j-1,word[1:])) or (j < self.width-1 and self.dfs(i, j+1,word[1:])):
                return True
            # 路径不通，恢复当前点的字符
            self.board[i][j] = word[0]
        return False

if __name__ == "__main__":
    board =[  ['A','B','C','E'],  ['S','F','C','S'],  ['A','D','E','E']]
    print(Solution().exist(board, 'SEE'))

