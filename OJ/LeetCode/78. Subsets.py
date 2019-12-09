class Solution:
    def subsets(self, nums):
        res = []
        l = len(nums)
        self.dfs(sorted(nums), 0, [], res, l)
        return res
    
    def dfs(self, nums, index, path, res, l):
        res.append(path)
        for i in range(index, l):
            self.dfs(nums, i+1, path+[nums[i]], res, l)


if __name__ == "__main__":
    print(Solution().subsets([1,2,3]))