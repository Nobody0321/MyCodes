class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if rowIndex <2:
            return [[1],[1,1]][rowIndex]

        rows = [[1],[1,1]]
        while len(rows) != rowIndex+1:
            newRow = [0] * (len(rows)+1)
            newRow[0] = newRow[-1] = 1
            for i in range(1, len(newRow)-1):
                newRow[i] = rows[-1][i-1] + rows[-1][i]
            rows.append(newRow)
        return rows[-1]

if __name__ == '__main__':
    print(Solution().getRow(4))