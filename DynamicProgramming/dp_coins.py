# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 06:23:27 2020

@author: 71020
"""

#dp problem about changes of coins
coins = [1, 3, 5]
s = 11 # sum of coins
A = [i for i in range(s+1)] 

def dp(coins, s):   
    for i in range(s+1):
        for j in range(len(coins)):
            if i >= coins[j] and A[i] > A[i-coins[j]]+1:                
                A[i] = A[i-coins[j]]+1
    return A
A = dp(coins, s)
B = A[::-1]
changes = []
temp = B[0]
for i in B:
    if temp > i:
        m = B.index(i) - B.index(temp)
        temp = i
        changes.append(m)
print(changes)

        