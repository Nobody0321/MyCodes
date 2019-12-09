# 题目要求复杂度logn,所以自然想到用二分
class Solution:
    def __init__(self):
        self.min = -1
        self.max = -1
    def searchRange(self, nums, target):
        def binary_search(nums, l, r, target):
            if l > r:
                return
            mid_ix = l + (r - l)//2
            mid = nums[mid_ix]
            if mid == target:
                self.min = self.min if self.min <= mid_ix else mid_ix
                self.max = self.max if self.max >= mid_ix else mid_ix
                binary_search(nums, l, mid_ix-1, target)
                binary_search(nums, mid_ix+1, r, target)
            elif mid < target:
                binary_search(nums, mid_ix+1, r, target)
            elif mid > target:
                binary_search(nums, l, mid_ix-1, target)

        binary_search(nums, 0, len(nums)-1, target)
        return [self.min, self.max]
       

if __name__=='__main__':
    print(Solution().searchRange([5,7,7,8,8,10],8))