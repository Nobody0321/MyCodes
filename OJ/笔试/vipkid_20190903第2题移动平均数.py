# 从第一行读入一组整数xs，第二行读入一个整数k，求xs序列中，每k项的平均值组成的新序列。

# 例如：xs=[1,2,3,4,5,6,7], k=3时，output=[2,3,4,5,6]

# 输入
# 第一行，空格分隔的n个整数。
# 第二行，一个整数k。

# 输出
# 空格分割的n-k+1个平均数，每个平均数保留两位小数。


# 样例输入
# 1 2 3 4 5 6 7
# 3
# 样例输出
# 2.00 3.00 4.00 5.00 6.00

# s = list(map(int, input().strip().split()))
s = list(map(int, "1 2 3 4 5 6 7".strip().split()))

# w = int(input().strip())
w = 3
ret = []
for i in range(len(s) - w + 1):
    a = sum(s[i:i+w])/w
    ret.append(a)
for a in ret:
     print("%.2f " % a, end='')
print()