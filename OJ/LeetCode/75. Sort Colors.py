class Solution:
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        # # 解法1 使用快排
        # self.quick_sort(nums, 0, len(nums)-1)
        # print(nums)

        # # 解法2 hashmap
        # d = {0:0,1:0,2:0}
        # for each in nums:
        #     d[each] += 1
        # n = [0]*d[0] + [1]*d[1]+[2]*d[2]
        # for i, num in enumerate(n):
        #     nums[i] = num
        

    def quick_sort(self, nums, start, end):
        if end - start <= 0:
            return 
        base = nums[start]
        i, j = start, end
        while i != j:
            while nums[j] >= base and j > i:
                j -= 1
            while nums[i] <= base and j > i:
                i += 1
            if i < j:
                nums[i], nums[j] = nums[j], nums[i]
            
        nums[start], nums[i] = nums[i], nums[start]
    
        self.quick_sort(nums, i+1,end)
        self.quick_sort(nums, start, i-1)



if __name__ == "__main__":
    Solution().sortColors([2,0,2,1,1,0])