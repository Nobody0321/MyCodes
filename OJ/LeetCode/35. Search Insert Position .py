class Solution:
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target in nums:
            return nums.index(target)
        lenth = len(nums)
        if target < nums[0]:
            return 0
        if target > nums[lenth-1]:
            return lenth
        for i in range(0,lenth):
            if nums[i] < target and nums[i+1] >target:
                return i+1

if __name__ == '__main__':
    s  = Solution()
    print(s.searchInsert([1,3,5,6],2))