def d2b(num):
    if num < 2:
        return num
    ret = 0
    c = 1
    while num:
        ret += c * (num % 2)
        num = num // 2
        c *= 10
    return ret

def ispal(num):
    r_num = 0
    c = 1
    while num:
        r_num += c*(num%10)
        num = num // 10
        c *= 10

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
    #     num = int(input())
    #     s = d2b(num)
    #     if ispal(s):
    #         print('YES')
    #     else:
    #         print('NO')
        num = int(input())
        b_num = bin(num)[2:]
        if b_num == b_num[::-1]:
            print('YES')
        else:
            print('NO')
