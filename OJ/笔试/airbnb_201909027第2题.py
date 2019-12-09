def divide(dividend, divisor):
    # Write your code here
    res = dividend/divisor
    if res == int(res):
        return str(int(res))
    res = str(res)
    dot_idx = res.index(".")
    remain = res[dot_idx+1:]
    repeat_start = 0
    repeat_s = None
    f = False
    for i in range(len(remain)):
        for j in range(i, len(remain)):
            s = remain[i:j+1]
            repeat_times = len(remain[i:]) // (j+1-i)
            if repeat_times <= 1:continue
            if s * repeat_times in remain:
                repeat_start = i
                repeat_s = s
                f = True
                break
        if f:
            break

    if repeat_s:
        remain = remain[:repeat_start] + "({})".format(repeat_s)
        return res[:dot_idx+1] + remain
    else:
        return res



print(divide(90, 45))