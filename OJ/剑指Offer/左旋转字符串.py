class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        # 循环移位，取长度的模即可
        if s == '':
            return s
        n = n % len(s)
        return s[n:]+s[:n]

if __name__ == "__main__":
    print(Solution().LeftRotateString('abcdefg',2))