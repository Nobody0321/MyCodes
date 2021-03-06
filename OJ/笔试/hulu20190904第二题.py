# 时间限制：C/C++语言 1000MS；其他语言 3000MS
# 内存限制：C/C++语言 131072KB；其他语言 655360KB
# 题目描述：
# Hulu有一些列的视频文件，每个文件都有对应的码率，为整数。输入长度为 n 的视频码率数组 arr ，现在定义两个文件区段之间最大码率为：

#             p[i][j] = max(arr[i], arr[i+1], ... , arr[j]), 0 <= i <= j <= n-1.

# 针对所有满足条件 0 <= i <= j <= n-1 的 (i,j) 对，求 p[i][j] 的总和.

# 输入
# 第一行为 n，表示数组的长度。第二行为空格分开的 n 个码率.

# 输入满足： 1 <= n <= 1000000， 1 <= arr[i] <= 1000000，arr中可能存在重复值.

# 输出
# 输出一个数字，即 p[i][j] 的总和，如果总和超过 1000000007 ，则返回对1000000007取模的结果.


# 样例输入
# 3
# 1 2 2
# 样例输出
# 11

# 提示
# 解释：满足要求的 p[0][0] = 1, p[0][1] = 2, p[0][2] = 2, p[1][1] = 2, p[1][2] = 2, p[2][2] = 2. 将这些相加，结果为 11.
n = int(input())
s = list(map(int, input().strip().split()))
i, j = 0, n-1
ret = 0
t_max = max(s)
while i != j:
    ret += t_max
    while i!= j and s[j] < t_max:
        ret += t_max
        j -= 1
    while i!= j and s[i] < t_max:
        ret += t_max
        i += 1