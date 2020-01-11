class Solution:
    # def firstMissingPositive(self, nums):
    #     """simplest method"""
    #     if not nums or len(nums) == 0:
    #         return 1
    #     nums = list(set(nums))
    #     nums.sort()
    #     for i in range(len(nums)):
    #         if nums[i] > 0:
    #             nums = nums[i:]
    #             break
    #     if nums[0] > 1:
    #         return 1
    #     else:
    #         for i in range(len(nums)):
    #             if nums[i] != i + 1:
    #                 return i + 1
    #         return nums[-1] + 1

    def firstMissingPositive(self, nums):
        l = len(nums)
        for i in range(l):
            while nums[i] > 0 and nums[i] <= l and nums[nums[i] - 1] != nums[i]:
                    t = nums[i] - 1
                    nums[i], nums[t] = nums[t], nums[i]

        for i in range(l):
            if nums[i] != i + 1:
                return i + 1
        return l + 1

if __name__ == "__main__":
    nums = [3, 4, -1 ,]
    print(Solution().firstMissingPositive(nums))