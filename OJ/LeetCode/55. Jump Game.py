class Solution:
    def canJump(self, nums):
        # 贪心, 从0开始出发,记录能达到的最远点
        reachable = 0
        for ii,num in enumerate(nums):
            if ii <= reachable:
                reachable = max(reachable, ii + num)
            else:
                return False
        return True


if __name__ == "__main__":
    print(Solution().canJump([2,3,1,1,4]))