class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        t = 1 if x > 0 else -1
        x = x if x > 0 else -x
        y = 0
        while x > 0:
            res = x % 10
            y = y * 10 + res
            x = x // 10
        y = t * y
        if  y <= 2 ** 31 - 1 and y >= -2 ** 31:
            return y
        else:
            return 0

        
if __name__ == '__main__':
    s = Solution()
    print(s.reverse(123))