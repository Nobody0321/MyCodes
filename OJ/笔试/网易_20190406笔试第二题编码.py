def D2S(s):
    n = int(s)
    if n < 10:
        return str(n)
    else:
        return chr(n-10+ord('A'))

def B2D(str_n):
    t = 0
    c = 1
    n = int(str_n)
    while n > 0:
        r = n %10
        t += r * c
        c *= 2
        n = n // 10
    return str(t)

def D2B(str_n):
    n = int(str_n)
    t = ''
    while n > 0:
        r = str(n % 2)
        t = r + t
        n = n //2
    return ('0000000000' + t)[-10:]

# T = int(input())
T = 1
res = []
for i in range(T):
    # nums = input()
    nums = '5555'
    # step 1 divide into 3-group, pad starting '0'
    split_nums = []
    t = ''
    l = len(nums)
    for j in range(l):
        t = nums[l-j-1] + t
        if len(t) == 3:
            split_nums.append(t)
            t = ''
        if j==l-1:
            t = '000'+t
            t = t[-3:]
            split_nums.append(t)
    split_nums.reverse()

    # step2 convert to Binary, concat all, remove starting '0's
    split_nums = list(map(D2B,split_nums))
    joint_nums = ''.join(split_nums)
    for k in range(len(joint_nums)):
        if joint_nums[k] != '0':
            joint_nums = joint_nums[k:]
            break

    # step3 diveide into 5-group, pad '0'
    split_nums = []
    t = ''
    l = len(joint_nums)
    for j in range(l):
        t = joint_nums[l-1-j] + t
        if len(t) == 5:
            split_nums.append(t)
            t = ''
        if j==l-1 and t!= '':
            t = '00000'+t
            t = t[-5:]
            split_nums.append(t)
    split_nums.reverse()

    # step4 convert to 32-ary
    for j in range(len(split_nums)):
        split_nums[j] = D2S(B2D(split_nums[j]))
    # split_nums.reverse()
    res.append(''.join(split_nums))

for i in range(len(res)):
    print(res[i],end='')
    if i != len(res) -1:
        print()