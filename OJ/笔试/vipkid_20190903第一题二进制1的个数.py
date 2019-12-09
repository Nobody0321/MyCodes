# 输入一个大于0的整数，求其二进制表示中1的个数。例如，输入3，它的二进制表示为11，其中1的个数为2，输出2。

# 输入
# 输入一个大于0的整数
# 输出
# 二进制中1的个数

# 样例输入
# 3
# 样例输出
# 2

num = int(input().strip())
# def d2b(n):
#     c = 1
#     while n > 1:
#         t = n % 2
#         n = n // 2
#         c = c + t
#     c = 10 * c + n
#     return c

if __name__ == "__main__":
    n = bin(num)
    # n = 1010
    count = 0
    while n != 0:
        print(n)
        n = (n - 1) & n
        count += 1
    print(count)