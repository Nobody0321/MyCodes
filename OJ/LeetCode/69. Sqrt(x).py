class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0 or x == 1:
            return x
        for i in range(0, x+1):
            if i * i > x:
                return i - 1
            # elif i*i >x and (i-1) **2 <x:
            #     return i-1


if __name__ == '__main__':
    s = Solution()
    print(s.mySqrt(1441682837))