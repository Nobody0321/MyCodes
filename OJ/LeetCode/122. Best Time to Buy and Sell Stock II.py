# 不知道算是贪心还是什么，，先求最大的买卖日期，然后计算剔除买卖阶段所有日期之外的最大值
# 思路2: 遍历数组，计算所有第二天比前一天价格高的值，加起来就是最后的最大收益。


class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1:
            return 0
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] - prices[i-1] > 0:
                profit = profit + (prices[i]-prices[i-1])
        return profit