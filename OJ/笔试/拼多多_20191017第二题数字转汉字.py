def trans(num):
    unit = "拾佰仟"
    number = "壹贰叁肆伍陆柒捌玖"
    ret = ""
    count = 0
    while num:
        c = num % 10
        if c:
            c = number[c - 1]
            if count:
                c += unit[count - 1]
            count += 1
        else:
            c = ""
        ret = c + ret
        print(ret)
        num = num // 10
    return ret

if __name__ == "__main__":
    # n = int(input())
    n = 1234567
    ret = ""
    while n:
        t = n % 10000
        ret = trans(t) + "万" + ret
        n = n // 10000
    print(ret)
        