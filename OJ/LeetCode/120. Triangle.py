# 观察得，每一行的第i个点只能转移到下一行的第i个或者第i+1个点
# 然后运用dp即可
# 反过来dp会更快
class Solution:
    def minimumTotal(self, triangle):
        if len(triangle) == 1:
            return triangle[0][0]

        for level_n in range(len(triangle)-2, -1,-1):
            for n in range(len(triangle[level_n])):
                triangle[level_n][n] += min(triangle[level_n+1][n], triangle[level_n+1][n+1])

        return triangle[0][0]        
        