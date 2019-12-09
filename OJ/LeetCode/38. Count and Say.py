class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        res = '1'
        for i in range(n-1):
            new_res = ''
            j = 0
            while j < len(res):
                count = 1
                while j < len(res)-1 and res[j] == res[j+1]:
                    j = j + 1
                    count = count + 1
                j = j + 1
                new_res = new_res + str(count) + res[j-1]
            res = new_res
        return res


if __name__ == '__main__':
    s = Solution()
    print(s.countAndSay(10))