class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        digits = list(map(str, digits))
        n = ''.join(digits)
        n = str(int(n)+1)
        lenth = len(n)
        new_digits = []
        for i in range(len(n)):
            new_digits.append(int(n[lenth - 1 -i]))
        new_digits.reverse()  # 这个函数没有返回值
        return new_digits


if __name__ == '__main__':
    s = Solution()
    print(s.plusOne([1,2,3]))
