class Solution:
    def longestCommonPrefix(self, strs):
        if len(strs) == 1:
            return strs[0]
        if len(strs) == 0:
            return ""
        
        #1. find the shortest str
        min_l = 200
        for each in strs:
             min_l = min(len(each), min_l)
        if min_l == 0:
            return ""
            
        # 从小到大遍历，如果当前不行，那更大的子串也不行，返回上一个字串就行
        for i in range(1, min_l + 1):
            compare = strs[0][:i]
            for each in strs:
                if each[:i] != compare:
                    return compare[:i-1]
        # 如果都行，那就返回 min_l
        return strs[0][:min_l]

if __name__ == '__main__':
    s = Solution()
    print(s.longestCommonPrefix(["a", "ab"]))