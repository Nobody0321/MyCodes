# 思路 
# 不论如何，target肯定是在一个小的有序串中，
# 所以我们只要能确定包含target的最大有序子串范围，就能确定target
class Solution:
    def search(self, nums, target):
        l, r = 0, len(nums) - 1 
        
        while l <= r:
            # 这个等号，在要查找的数存在的情况下，可以最终定位到mid
            mid = (l + r) // 2

            if target == nums[mid]:
                return mid

            if nums[l] <= nums[mid]:
                # the left part is sorted
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                # the right part is sorted
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1

        return -1

if __name__ == "__main__":
    print(Solution().search([4,5,6,7,0,1,2],0))