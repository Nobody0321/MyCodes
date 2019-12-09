class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if len(s):
            import collections
            s = list(s)
            d = collections.OrderedDict()
            for i in range(len(s)):
                if s[i] not in d.keys():
                    d[s[i]] = i
                else:
                    d[s[i]] = -1
                    
            for each in d.values():
                if each != -1:
                    return each
        return -1
        
if __name__ == "__main__":
    s = 'google'
    print(Solution().FirstNotRepeatingChar(s))