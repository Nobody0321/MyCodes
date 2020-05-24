class Solution:
    def trap_1(self, height):
        """暴力解法
        
        Args:
            height ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        ans = 0
        for i in range(len(height)):
            max_l = max_r = height[i]
            for j in range(0, i):
                max_l = max(max_l, height[j])
            for j in range(i + 1, len(height)):
                max_r = max(max_r, height[j])
            ans += min(max_r, max_l) - height[i]
        return ans


if __name__ == "__main__":
    print(Solution().trap_1([0,1,0,2,1,0,1,3,2,1,2,1]))