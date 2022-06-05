class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        l1, l2 = len(nums1), len(nums2)

        # 确保nums2更大，保证右边界可以缩到nums2中，这样后边的j才有意义（j>0）
        if l1 > l2:
            return self.findMedianSortedArrays(nums2, nums1)
        
        left, right = 0, l1
        # 使用二分法，在nums1、nums2 中找到 中位边界
        while (left <= right):
            # 初始为：小数组的中位数的位置,
            # 左边数组中，大于整体中位数的最小数index
            i = (left + right) // 2
            # 右边数组中，大于整体中位数的最小数index，
            # 保证i+j == ceil((l1+l2) / 2)
            j = (l1 + l2 + 1) // 2 - i

            # nums1, nums2 中的左右边界
            left_1 = float(-10**6) if i == 0 else nums1[i - 1]
            right_1 = float(10**6) if i == l1 else nums1[i]

            left_2 = float(-10**6) if j == 0 else nums2[j - 1]
            right_2 = float(10**6) if j == l2 else nums2[j]

            # 窗口宽度固定是 (l1 + l2 + 1) // 2，
            # 如果满足左边都不大于右边，说明这个窗口找到了，可以退出循环了
            # 分别是nums1[:i] + nums2[:j]
            if left_1 <= right_2 and left_2 <= right_1:
                # 总长度为奇数的话，小于中位数的窗口偏大，所以找左半边的最大值
                if (l1 + l2) % 2 == 1:
                    return max(left_1, left_2)
                # 总长度为偶数，找中间边界两旁的数求平均
                else:
                    return (max(left_1, left_2) + min(right_1, right_2)) / 2
            # 不满足结束条件就继续循环
            else:
                # 整体应该向左（小）寻找
                if left_1 > right_2:
                    right -= 1
                # 整体应该向右（大）寻找
                else:
                    left += 1

        return 0.0


if __name__ == "__main__":
    nums1 = [1, 3]
    nums2 = [2]
    print(Solution().findMedianSortedArrays(nums1, nums2))

    nums1 = [1, 2]
    nums2 = [3, 4]
    print(Solution().findMedianSortedArrays(nums1, nums2))