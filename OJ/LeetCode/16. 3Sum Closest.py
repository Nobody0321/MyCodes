class Solution:
    def threeSumClosest(self, nums, target):
        nums.sort()
        result = 1001
        for i in range(len(nums)):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                sums = nums[i] + nums[j] + nums[k]
                if abs(sums - target) < abs(result - target):
                    result = sums
                if sums == target:
                    return target
                elif sums > target:
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                            k -= 1
                elif sums < target:
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                            j += 1
        return result


if __name__ == "__main__":
    nums = [0,0,0]
    target = 0
    print(Solution().threeSumClosest(nums, target))
