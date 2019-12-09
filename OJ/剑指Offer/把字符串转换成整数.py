class Solution:
    def StrToInt(self, s):
        ret = 0
        flag = 1
        if not s:
            return 0
        if type(s) == type(ret):
            return s
        s = list(s)
        if s[0] == '+' or s[0] == '-': 
            flag = -1 if s[0] == '-'
            s = s[1:]
        for c in s:
            val = ord(c) - ord('0')
            if val >=10 or val <0:
                return 0
            ret = ret * 10 + val
        return ret * flag