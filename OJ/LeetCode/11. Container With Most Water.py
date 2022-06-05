class Solution:
    def maxArea(self, height):
        max_a = 0
        i, j = 0, len(height) - 1
        # 从最宽的开始算，逐渐缩小，哪边矮就换哪边
        while i < j:
            max_a = max(max_a, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_a

if __name__ == "__main__":
    print(Solution().maxArea([1,8,6,2,5,4,8,3,7]))