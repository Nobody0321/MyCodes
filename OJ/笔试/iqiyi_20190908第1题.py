# 给定一个长度为N-1且只包含0和1的序列A1到AN-1，如果一个1到N的排列P1到PN满足对于1≤i<N，当Ai=0时Pi<Pi+1，当Ai=1时Pi>Pi+1，则称该排列符合要求，那么有多少个符合要求的排列？

# 输入
# 第一行包含一个整数N，1<N≤1000。

# 第二行包含N-1个空格隔开的整数A1到AN-1，0≤Ai≤1

# 输出
# 输出符合要求的排列个数对109+7取模后的结果。


# 样例输入
# 4
# 1 1 0
# 样例输出
# 3

# 思路： 有点类似全排列的回溯
# N = int(input())
# A = list(map(int, input().split()))
N = 4
A = [1,1,0]
s = [i for i in range(1, N+1)]
count = [0]

def count_n(s, n, pre):
    if n == N-1:
        # 终止条件
        count[0] += 1
        return
    else:
        if len(s) <= 1:
            return
        for i in range(len(s)):
            if A[n] == 0 and s[i] > pre:
                count_n(s[:i]+s[i+1:], n+1, s[i])
            if A[n] == 1 and s[i] < pre:
                count_n(s[:i]+s[i+1:], n+1, s[i])

for i in range(len(s)):
    count_n(s[:i]+s[i+1:], 1, s[i])
        

print(count[0])
