# 这题主要是没有看清楚题目要求，要求这个函数没有返回值，直接修改nums1就行

class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        temp = []
        i,j = 0,0
        while(i < m and j < n):
            if nums1[i] < nums2[j]:
                temp.append(nums1[i])
                i = i+1
            else:
                temp.append(nums2[j])
                j =j+1
        while(i < m):
            temp.append(nums1[i])
            i = i+1
        while(j < n):
                temp.append(nums2[j])
                j = j+1
        for i in range(len(temp)):
            nums1[i] = temp[i]
        return nums1


if __name__== "__main__":
    s = Solution()
    print(s.merge([1,2,3,0,0,0],3,[2,5,6],3))