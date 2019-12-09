class Solution:
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s
        
        res = [''] * numRows  # res 存放所有行
        index, step = 0, 1  # index 指向当前转到的行，step控制index的增减
        for c in s:
            res[index] += c
            if index == 0:
                # index从0开始，是递增方向
                step = 1
            elif index == numRows - 1:
                # index 到numRows, 是递减方向
                step = -1
            index += step

        return ''.join(res)  # 连接所有行      


if __name__ == "__main__":
    print(Solution().convert( s = "PAYPALISHIRING", numRows = 4))