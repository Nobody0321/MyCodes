class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        flag = 1 if x >= 0 else -1
        x = str(abs(x))
        x = flag * int(x[::-1])
        if  x <= 2 ** 31 - 1 and x >= - 2 ** 31:
            return x
        else: return 0

        
if __name__ == '__main__':
    s = Solution()
    print(s.reverse(-123))