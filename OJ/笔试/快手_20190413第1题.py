# 判断长度差几个
# 判断短的有几个跟长的不一样

s1 = input()
s2 = input()
l1 = len(s1) + 1
l2 = len(s2) + 1
pre = [0]*12
for j in range(l2):
    pre[j] = j
for i in range(1, l1):
    cur = [i]*l2
    for j in range(1, l2):
        cur[j] = min(cur[j-1]+1, pre[j]+1, pre[j-1]+(s1[i-1]!=s2[j-1]))
    pre = cur[:]
print(pre[-1])