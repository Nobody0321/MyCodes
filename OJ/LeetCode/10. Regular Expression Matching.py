class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if s == "":
            return p is ""
        if p == "":
            return False
        if len(p) > 1 and p[1] == "*":
            if s[0] == p[0] or p[0] == ".":
                return self.isMatch(s[1:], p) or self.isMatch(s, p[1:])

    def isEmpty(self, p):
        if len(p): return False
        for i in range(len(p)):
            