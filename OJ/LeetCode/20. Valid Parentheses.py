class Solution:
    # 核心思想是实现一个栈
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        v = []
        start = ['(','[','{']
        for i in range(len(s)):
            if s[i] in start:
                v.append(s[i])
                continue
            elif s[i] == ')':
                if len(v) == 0 or v.pop() != '(':
                    return  False
            elif s[i] == ']':
                if len(v) == 0 or v.pop() != '[':
                    return False
            elif s[i] == '}':
                if len(v) == 0 or v.pop() != '{':
                    return False
        return len(v) == 0

if __name__ == '__main__':
    s = Solution()
    print(s.isValid(r'(){}'))