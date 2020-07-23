# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:12:36 2020

@author: 71020
"""


def dp(x: list) -> int:
    s1, s2 = x[0], x[1]
    d = [[0 for col in range(len(s2)+1)] for row in range(len(s1)+1)]
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                d[i][j] = d[i-1][j-1] + 1
            else:
                d[i][j] = max(d[i-1][j], d[i][j-1])
    i = len(s1)
    j = len(s2)
    pa = []
    while d[i][j]:
        if s1[i-1] == s2[j-1]:
            pa.append(s1[i-1])
            i -= 1
            j -= 1
        elif s1[i-1] != s2[j-1] and d[i-1][j] > d[i][j-1]:
            i -= 1
        else: 
            j -= 1
    pa.reverse()
    return d[-1][-1], ''.join(pa)

    
    
if __name__ == "__main__":
    data = [
        'abcfbc abfcab\n',
        'programming contest\n',
        'abcd mnp'
        ]
    for _ in data:
        data_i = _.strip().split(" ")
        res, p = dp(data_i)
        print(res, "lcs:", p)

        
    
    
    
    
    
    
    
    
    
    
    
    
        