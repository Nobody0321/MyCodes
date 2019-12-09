class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        l = len(numbers)>>1
        for each in numbers:
            if numbers.count(each) > l:
                return each
        return 0