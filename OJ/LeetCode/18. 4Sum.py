class Solution:
    """
    把 Nsum 都简化为2sum (N >= 2), N sum 的复杂度是 
    """
    def fourSum(self, nums, target):
        nums.sort()
        results = []
        self.find_n_sum(nums=nums, target=target, n=4, results=results, path=[])
        return results
        
    def find_n_sum(self, nums, target, n, results, path=[]):
        # 跳出条件
        # 1. N 比数组大
        # 2. N <= 1 没必要比了
        # 3. target 比nums 中最小的* N 都小，或者比nums 中最大的 * N 都大
        if len(nums) < n or n <= 1 or target < nums[0]* n or target > nums[-1] * n:
            return

        # 成功退化到2sum，就开始正常计算
        if n == 2:
            l, r = 0, len(nums) - 1
            while l < r:
                sums = nums[l] + nums[r]
                if sums == target:
                    results.append(path + [nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif sums > target:
                    r -= 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif sums < target:
                    l += 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
        elif n != 2:
            # 不需要最后N-1个点 当作第一个点的结果，因为肯定答案长度不够，无法满足N sum
            for i in range(len(nums) - (n - 1)):
                if i == 0 or (i >= 1 and nums[i] != nums[i - 1]):
                    self.find_n_sum(nums=nums[i+1:], target=target-nums[i], n=n-1, results=results, path=path+[nums[i]])
        return

if __name__ == "__main__":
    print(Solution().fourSum([1,0,-1,0,-2,2], 0))