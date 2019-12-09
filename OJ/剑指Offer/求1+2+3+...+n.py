class Solution:
    def Sum_Solution(self, n):
        # write code here
        if n <= 0:
            return n
        else:
            return n + self.Sum_Solution(n-1)