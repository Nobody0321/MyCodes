# class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         # 递归，超时了
#         def is_ix_match(i, j):
#             if i == len(s) and j == len(p):
#                 return True

#             if j == len(p):
#                 return False

#             if i == len(s):
#                 return is_ix_match(i, j + 1) if p[j] == '*' else False
 
#             if p[j] == "*":
#                 return is_ix_match(i + 1, j) or is_ix_match(i, j + 1)

#             if s[i] == p[j] or p[j] == "?":
#                 return is_ix_match(i + 1, j + 1)

#             return False
            
#         return is_ix_match(0, 0)
    
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # 试试DP
        m, n = len(s), len(p)
        # m 行 n 列
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        # 如果 s 是“”，p如果非空且全是*即可
        for j in range(1, n):
            if p[j - 1] == "*":
                dp[0][j] = dp[0][j - 1] 
            
        for i in range(1, m):
            for j in range(1, n):
                # 如果pattern 上一个是*，那么如果不要*还可以匹配，或者当前patern可以跟上一个s 的字符匹配（把*看作匹配空字符），说明也可以，
                if p[j - 1] == "*":
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                # 如果pattern 是其他符号，那么如果之前可以匹配，并且现在可以匹配，就是true
                elif s[i] == p[j] or p[j - 1] == "?":
                    dp[i][j] = dp[i - 1][j - 1]
        print(dp)
        return dp[m-1][n-1]
    

if __name__ == "__main__":
    s = "aa"
    p = "*"
    print(Solution().isMatch(s, p))