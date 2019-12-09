class Solution:
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        needle_len = len(needle)
        print(needle_len)
        lenth = len(haystack)
        print(lenth)
        flag = 0
        if len(haystack) == 0:
            if haystack == needle:
                return 0
        for i in range(0,lenth):
            if haystack[i:i + needle_len] == needle:
                flag = 1
            if flag == 1:
                break
        if flag == 1:
            return i
        else:
            return -1

if __name__ == '__main__':
    s = Solution()
    print(s.strStr('',''))