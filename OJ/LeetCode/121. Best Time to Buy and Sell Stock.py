class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0:
            return 0
        minPrice = prices[0]
        maxProfit = 0
        for price in prices:
            minPrice = min(price, minPrice)
            maxProfit = max(maxProfit, price - minPrice)
        return maxProfit