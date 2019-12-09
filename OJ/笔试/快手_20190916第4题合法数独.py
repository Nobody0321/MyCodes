def isValidSudoku(mat):
    big = []
    for i in range(0,9):
        for j in range(0,9):
            if mat[i][j] != 'X':
                cur = mat[i][j]
                if (i,cur) in big or (cur,j) in big or (i//3,j//3,cur) in big:
                    return False
                big.append((i,cur))
                big.append((cur,j))
                big.append((i//3,j//3,cur))
    return True

# mat = []
# for i in range(9):
#     mat.append(input().strip())
mat = ['53XX7XXXX', '6XX195XXX', 'X98XXXX6X', '8XXX6XXX3', '4XX8X3XX1', '7XXX2XXX6', 'X6XXXX28X', 'XXX419XX5', 'XXXX8XX79']
print(isValidSudoku(mat))