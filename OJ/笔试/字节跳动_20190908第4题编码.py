# 有点类似有效ip，回溯法

# nums = input().strip()
nums = '12'
cur = ''
ret = []


def DECODE(nums, cur_result):
    if len(nums) == 0:
        ret.append(cur_result)
    t = ''
    for i in range(len(nums)):
        t += nums[i]
        if int(t) < 27:
            DECODE(nums[i+1:], cur_result + chr(int(t)-1 + ord('A')))
        else:
            t = nums[i]
            DECODE(nums[i+1:], cur_result + chr(int(t)-1 + ord('A')))


DECODE(nums, '')
for i in range(len(ret)):
    print(ret[i], end='')
    if i != len(ret)-1:
        print()
