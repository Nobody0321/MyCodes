class Solution:
    # 贪心算法，只考虑局部最优解
    # Sum代表nums[i]之前字串和（不一定是从头开始），如果Sum<0那必然会拉低nums[i]开始的子串和，
    # 那么就舍弃（即置Sum = 0），从nums[i]开始重新求和
    # Max用于保存Sum的最大值
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        Sum = 0
        Max = -2**32
        for i in range(len(nums)):
            Sum = nums[i] + (Sum if Sum > 0 else 0)
            Max = Sum if Sum > Max else Max
        return Max   



if __name__ == "__main__":
    s = Solution()
    print(s.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))