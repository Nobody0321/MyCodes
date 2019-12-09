class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """

        if len(strs) == 0:
            return ''
        if len(strs) == 1:
            return strs[0]
        i = 1
        while True:
            common = strs[0][0:i]
            for each in strs:
                if common != each[0:i]:
                    return common[0:-1]
            i = i + 1
        return common
        

if __name__ == '__main__':
    s = Solution()
    print(s.longestCommonPrefix(["","",""]))