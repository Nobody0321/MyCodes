class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        解题思路，给定的数组已排好序了，
        从j= 0出发，遇到的每一个不与nums[j]相同的数字都放在j后面，j= j+1
        最后0~j-1放了所有不同的数字
        """
        if len(nums) == 0:
            return 0
        else:
            j = 0
            for i in range(0, len(nums)):
                if nums[i] != nums[j]:
                    nums[i], nums[j+1] = nums[j+1], nums[i]
                    j = j + 1
        return j + 1

if __name__ == "__main__":
    s = Solution()
    while True:
        try:

            n = input('请输入:')
            n = list(n)
            nums = []
            for each in n:
                nums.append(int(each))
            print(s.removeDuplicates(nums))
        except:
            break
