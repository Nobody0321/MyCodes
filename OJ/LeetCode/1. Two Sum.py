class Solution:
    # python中dict的检索方式是哈希表，最多O(n)，最少O(1)，快于O(n^2)
    # python用O(n^2)会超时
    def twoSum(self, nums, target):
        dic = {}
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return (dic[target-nums[i]], i)
            dic[nums[i]] = i

if __name__ == '__main__':
    print(Solution().twoSum([3,2,4],6))