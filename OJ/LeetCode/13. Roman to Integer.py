class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        dic = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000,
        }
        lenth = len(s)
        sum = dic.get(s[0])
        if lenth ==1:
            return sum
        for i in range(1,lenth):
            pre = dic.get(s[i-1])
            cur = dic.get(s[i])
            if cur <= pre:
                sum += cur
            else:
                sum = sum - 2 * pre + cur
        return sum


if __name__ == '__main__':
    s = Solution()
    print(s.romanToInt('XI'))