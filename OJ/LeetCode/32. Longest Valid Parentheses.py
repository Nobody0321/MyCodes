# 思路：
# 还是得用栈，每次遇到左括号入栈，遇到右括号，合法的话将入栈的左括号替换成当前最大有效长度
# 所以栈初始化为有一个元素0
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        l = 0
        stack = [0]
        for c in s:
            if c == '(':
                stack.append(0)
            else:
                if len(stack) > 1:
                    val = stack.pop()
                    # stack.pop()
                    stack[-1]+= val + 2
                    l = max(l, stack[-1])
                else:
                    stack = [0]
        return l