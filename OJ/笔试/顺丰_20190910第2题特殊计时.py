# 题目描述：
# 24小时计时制是一个广为使用的计时体系。但是不同地方使用的计数进制是不同的，例如，在一个古老的村庄就是使用二进制下的24小时制，这时“11：11”表示的就是3点03分。

# 现在给出一个未知的时刻，用形如“a:b”的形式来表示，a，b分别是一个字符串，字符串可以由0-9和A-Z组成，分别代表0-9和10-35。请你求出这个时刻所处的所有可能的进制。

# 输入
# 输入仅包含一行，即a:b的形式，a，b的含义及组成如题面所示

# 输出
# 输出可以包含若干个整数，如果不存在任何一个进制符合要求，则输出“-1”，如果有无穷多的进制数符合条件，则输出“0”，否则按从小到大的顺序输出所有进制数，中间用空格隔开


# 样例输入
# 00002:00130
# 样例输出
# 4 5 6

# 思路，不论是几进制，前面的数不能大于23， 后边的不能大于59

def convert(num, d):
    """
    d进制转10进制
    """
    c = 1
    ret = 0
    while num:
       ret += (num % 10) * c
       c *= d
       num //= 10
    return ret


def parseABC(s, d):
    """
    字符串转10进制
    """
    num = 0
    dd = 1
    for c in s[::-1]:
        v = ord(c)
        if ord('A') <= v <= ord('Z'):
            num += ((v - ord('A') + 10) * dd)
        else:
            num += (int(c) * dd)
        dd *= d
    return num


if __name__ == "__main__":
    # a,b = input().split(':')
    a,b = "00002:00130".split(':')
    result = []
    min_n = 0
    
    for c in a+b:
        v = ord(c)
        if ord('A') <= v <= ord('Z'):
            min_n = max(min_n, v - ord('A') + 10)
        else:
            min_n = max(min_n, int(c))

    i = min_n + 1

    while True:
        t_a, t_b = parseABC(a, i), parseABC(b, i)
        t_a, t_b = convert(t_a, i), convert(t_b, i)

        if t_a > 23 or t_b > 59:
            break

        result.append(i)
        i += 1
    if result == []:
        print(-1)
    else:
        print(result[::-1])

