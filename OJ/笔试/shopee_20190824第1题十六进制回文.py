import sys
if __name__ == "__main__":
    toG = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E','F']
    def D2G(num):
        if num < 16:
            return toG[num]
        else:
            ret = ''
            while num >= 16:
                n =  toG[num % 16]
                num = num // 16
                ret = str(n) + ret
            num = toG[num]
            ret = str(num) + ret
            return ret

    # 读取第一行的n
    # n = int(sys.stdin.readline().strip())
    n = 282
    ret = D2G(n)
    print(ret)
    if ret == ret[::-1]:
        print(1)
    else:
        print(0)
