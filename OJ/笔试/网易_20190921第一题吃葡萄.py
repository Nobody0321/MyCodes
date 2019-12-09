# 解三元三次方程组
def calc_xyz(a, b, c):
    x = (1/2) * (a + b - c)
    y = (1/2) * (b + c - a)
    z = (1/2) * (a + c - b)
    return x, y, z

import math
T = int(input())
for _ in range(T):
    a, b, c = list(map(int, input().split()))
    print(math.floor(max(calc_xyz(a, b, c)))
