class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if len(array):
            total, ret = array[0], array[0]
            for i in range(1,len(array)):
                if total >= 0:
                    total += array[i]
                else:
                    total = array[i]
                if total > ret:
                    ret = total
            return ret
        return 0