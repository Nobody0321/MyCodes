class Solution:
    def maxSlidingWindow(self, nums, k):
        ret = []
        q = []
        for i in range(len(nums)):
            if q != [] and q[0] == i - k:
                # pop number outside of window
                q.pop(0)
            while q != [] and nums[q[-1]] < nums[i]:
                # pop number index less than nums[i]
                # so q only stores the greatest number's index in window
                q.pop()
            q.append(i)
            if i >= k - 1:
                # time to cal the ,max value
                ret.append(nums[q[0]])
        return ret


if __name__ == "__main__":
    print(Solution().maxSlidingWindow([1,3,1,2,0,5], 3))