# 解题思路： 
# 观察可知，二叉搜索树是从左下到根到右下顺序排序的，
# 即一个bst中序遍历是一个递增序列

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
        elif len(nums) == 1:
            return TreeNode(nums[0])
        mid = len(nums)//2
        root = TreeNode(nums[mid])
        nums1 = nums[:mid]
        nums2 = nums[mid+1:]
        root.left = self.sortedArrayToBST(nums1)
        root.right = self.sortedArrayToBST(nums2)
        return root

if __name__== '__main__':
    testcase = [-10,-3,0,5,9]
    s = Solution()
    print(s.sortedArrayToBST(testcase))