class Solution:
    def search(self, nums, target):
        # 旋转数组的查找，主要就是不断缩小查找范围，以及判断轴心在哪里
        # 主要是这个数组中有重复数字，所所以需要先对mid预处理
        if nums == []:
            return False

        start, end = 0, len(nums)-1
        while end >= start:
            while start < end and nums[start] == nums[end]:#这样的目的是为了能准确判断mid位置，所以算法的最坏时间复杂度为O(n)
                start += 1    
            mid = (end + start) //2
            if nums[mid] == target:
                return True
            elif nums[start] <= nums[mid]:
                # 说明左半部分是递增的
                if nums[mid] > target >= nums[start]:
                    end = mid - 1
                else:
                    start = mid + 1 
            elif  nums[mid+1] <= nums[end]:
                # 右半部分是有序的
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                return False
        
        return False

if __name__ == "__main__":
    print(Solution().search(nums =[1,3], target = 3))