class Solution:
    def fourSum(self, nums, target):
        nums.sort()
        results = []
        self.findNsum(nums, target, 4, [], results)
        return results
    
    def findNsum(self, nums, target, N, path, results):
        if len(nums) < N or N < 2 or target < nums[0]*N or target > nums[-1] * N: return
            # 第三个条件说明后面的数加起来肯定都大于target，
            # 第四个条件说明后面所有数加起来肯定小于target
            # 没必要在当前位置浪费时间了， 直接跳过

        if N == 2:
            # N == 2的时候问题可以简化为2Sum
            l, r = 0, len(nums) - 1
            while l< r:
                s = nums[l] + nums[r]
                if s == target:
                    # 当前已找到一个答案，再移动指针找下一个答案
                    results.append(path + [nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                    while l < r and nums[r] == nums[r+1]:
                        r -= 1
                elif s < target:
                    l += 1
                else:
                    r -= 1

        else:
            for i in range(len(nums)-N + 1):
                # 不需要遍历最后N-1个点，因为肯定答案长度不够，无法满足Nsum

                if i == 0 or i > 0 and nums[i-1] != nums[i]:
                    self.findNsum(nums[i+1:], target-nums[i], N-1, path+[nums[i]], results)
        return

if __name__ == "__main__":
    print(Solution().fourSum([1,0,-1,0,-2,2], 0))