# 在从左往右书写正确答案的时候，我们发现：
# 1.左括号的个数总是不小于右括号的个数，
# 2.在左括号小于n的时候，可以选择加左括号或者加右括号
# 3.在左括号数小于右括号的时候，只能选择
# 4.右括号数等于n，说明已经得到了一个满足条件的答案
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        return self.dfs(n,n,"",[])
    
    def dfs(self, left, right, s, result):
        if left>0:
            self.dfs(left-1, right,s+'(',result)
        if right > 0 and left < right:
            self.dfs(left,right-1,s+')',result)
        if right == 0:
            result.append(s)
        return result