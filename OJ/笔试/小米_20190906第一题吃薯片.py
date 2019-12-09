num = list(map(int, input().split()))
length = len(num)
dp = [[0]*length]*length
l = 0
if length % 2:
    for j in range(length):
        dp[j][j] = num[j]
    l = 2
else:
    for j in range(length-1):
        dp[j][j+1] = abs(num[j]- num[j+1])
    l = 3

idx = l
for k in range(idx, length, 2):
    for t in range(length-k):
        dp[t][t+l]=max(min(num[t]-num[t+1]+dp[t+2][t+l], num[t]-num[t+l]+dp[t+1][t+l-1]),
                             min(num[t+l]-num[t]+dp[t+1][t+l-1], num[t+l]-num[t+l-1]+dp[t][t+l-2]))
    l+=2

if(dp[0][-1]>=0):
        print("Yes")
else:
        print("No")