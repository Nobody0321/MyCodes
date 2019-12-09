class Solution:
    def removeDuplicates(self, nums):
        if nums == []:
            return 0
        # 思路： 前后指针，一个遍历数组，一个保存个数，额外用一个标志记录次数
        idx = 1
        flag = 1
        for i in range(1,len(nums)):
            # 判断数字是否重复
            if nums[i] == nums[i-1]:
                # 判断之前的出现次数，如果之前只出现过一次，就按顺序记录在数组中
                if flag == 1:
                    nums[idx] = nums[i]
                    idx += 1
                    flag = 2
            else:
                # 数字之前未曾出现过，直接记录，次数设置为1
                nums[idx] = nums[i]
                idx += 1
                flag = 1
        return idx



if __name__ == "__main__":
    nums = [1,1,1,2,2,3]
    idx = Solution().removeDuplicates(nums)
    print(nums[:idx+1])