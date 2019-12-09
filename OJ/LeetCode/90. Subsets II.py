# 可以类比56题，不过那题没有重复元素，所以需要稍加判断
class Solution:
    def subsetsWithDup(self, nums):
        nums.sort()
        ret, cur = [[]], []
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i-1]:
                # 当前数字暂未重复，那么
                # 1.将当前数字作为一个子集[]+[nums[i]]
                # 2.将当前数字与之前所有结果结合，构造新的结果
                cur = [item + [nums[i]] for item in ret]
            else:
                # 如果当前数字重复，那当前数字再与ret中某些结果组合，
                # 就会造成重复，但是只与上一步重复数字所形成的结果（
                # 暂存在cur中）再进行结合，则不会造成重复
                cur = [item + [nums[i]] for item in cur]
            ret += cur
        return ret