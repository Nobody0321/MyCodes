class Solution:
    def GetLeastNumbers_Solution_0(self, tinput, k):
        # write code here
        if k <= len(tinput):
            return sorted(tinput)[:k]
        return []
    
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or k == 0:
            return []
        ret = []
        for each in tinput:
            if len(ret) < k:
                ret.append(each)
            else:
                ret.sort()
                if each < ret[-1]:
                    ret = ret[:-1] + [each]
        return sorted(ret)