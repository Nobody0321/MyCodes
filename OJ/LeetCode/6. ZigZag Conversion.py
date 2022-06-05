class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) <= 1 or numRows == 1:
            return s
        mod = numRows * 2 - 2
        num_columns = len(s) // mod * (numRows - 1) + (1 if len(s) % mod < numRows else len(s) % mod - numRows + 1)
        result = [""] * numRows
        
        i = 0
        cur_row = 0
        cur_column = 0
        flag = 1
        for c in s:
            print(cur_row, cur_column)
            print(cur_row * num_columns + cur_column)
            result[cur_row] += c
            if cur_row == numRows - 1:
                flag = -1
            elif cur_row == 0:
                flag = 1

            cur_row += flag
            if flag == -1:
                cur_column += 1

        return "".join(result)


if __name__ == "__main__":
    print(Solution().convert( s = "PAYPALISHIRING", numRows = 4))
    print(Solution().convert( s = "AB", numRows = 1))