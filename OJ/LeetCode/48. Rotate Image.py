class Solution:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        # # 解法1. 使用zip函数的解压缩，将matrix按列打包
        # for i, conlum in enumerate(zip(*matrix)):
        #     matrix[i] = list(conlum)[::-1]
        # return matrix


        # 解法2. 规规矩矩的inplace替换
        # 首先 对每一个元素横竖坐标互换,相当于沿左上到右下的对角线翻转, 这样每一行再逆序就是最终结果
        for i in range(len(matrix[0])):
            for j in range(i,len(matrix)):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # 下面每一行逆序,python有更简便的方法
        for i in range(len(matrix)):
            matrix[i] = matrix[i][::-1]
        

if __name__ == "__main__":
    matrix = [[ 1,2,3], [ 4,5,6], [7,8,9]]
    Solution().rotate(matrix)