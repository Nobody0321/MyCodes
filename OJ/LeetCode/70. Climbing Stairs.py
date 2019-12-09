# You are climbing a stair case. It takes n steps to reach to the top.
#
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

# 可以看作一个递归,但是递归超时了,还可以用动态规划
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        step = [1,1]
        if n == 1 or n == 2 or n == 0:
            return n
        else:
            for i in range(2, n+1):
                step.append(step[i-1] + step[i-2])

        return step[n]


if __name__ == '__main__':
    s = Solution()
    print(s.climbStairs(4))