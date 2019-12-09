class Solution:
    def lengthOfLastWord(self, s):
            """
            :type s: str
            :rtype: int
            """
            if len(s) ==0 or len(s.replace(' ','')) == 0:
                return 0

            strlist = s.split(' ')
            while strlist[-1] == '':
                strlist = strlist[:-1]
            return len(strlist[-1])

if __name__ == '__main__':
    s = Solution()
    print(s.lengthOfLastWord('a   b123 '))