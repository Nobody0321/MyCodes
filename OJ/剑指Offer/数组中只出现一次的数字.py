class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce_0(self, array):
        ret = {}
        for each in array:
            if each in ret.keys():
                ret.pop(each)
            else:
                ret[each] = 1
        return [a for a in ret.keys()]


    def FindNumsAppearOnce(self, array):
        count = 0
        for each in array:
            count ^= each
        idx = 0
        while (count & 1) == 0:
            count >>= 1
            idx += 1
        a = b = 0
        for i in array:
            if i>>idx &1:
                a ^= i
            else:
                b ^= i
        return [a, b]