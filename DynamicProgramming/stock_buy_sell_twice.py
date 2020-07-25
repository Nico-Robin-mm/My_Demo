# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:01:42 2020

@author: 71020
"""

prices = [1, 5 , 2 , 6 , 9 , 10 , 2]
# 分成两部分考虑
class BestProfit(object):
    
    def best2(self, prices):
        profit = 0
        for i in range(len(prices)):
            prices_l = prices[:i]
            prices_r = prices[i:]
            profit_l = self.best1(prices_l)
            profit_r = self.best1(prices_r)
            profit_lr = profit_l + profit_r
            if profit_lr > profit:
                profit = profit_lr
        return profit
    
    def best1(self, prices: list) -> int:
        if len(prices) < 2:
            return 0
        minPrice = prices[0]
        profit = prices[1] - prices[0]
        for i in range(2, len(prices)):
            if minPrice > prices[i-1]:
                minPrice = prices[i-1]
            if profit < prices[i] - minPrice:
                profit = prices[i] - minPrice
        return max(profit, 0)
    
best = BestProfit()
print(best.best2(prices))  



# dynamic programming
def dp(prices):
    # initial
    buy1, sell1, buy2, sell2 = float('-inf'), 0, float('-inf'), 0
    
    for i in range(len(prices)):
        buy1 = max(buy1, -prices[i])
        sell1 = max(sell1, buy1+prices[i])
        buy2 = max(buy2, sell1-prices[i])
        sell2 = max(sell2, buy2+prices[i])
    return sell2

print(dp(prices))












