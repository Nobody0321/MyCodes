class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        lenth = len(nums)
        j = 0
        for i in range(0,lenth):
            if nums[i] == val:
                continue
            nums[j] = nums[i]
            j = j + 1
        return j

if __name__ == '__main__':
    s = Solution()
    a = s.removeElement([3,2,2,3],3)
    print(a)