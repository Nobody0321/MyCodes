class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        nums.sort()
        m = -1
        for i in range(1, len(nums)):
            m = max(m, nums[i] - nums[i-1])
        return m