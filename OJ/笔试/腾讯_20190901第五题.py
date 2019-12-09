t,k = list(map(int, input().strip().split()))
A = []
B = []
for _ in range(t):
    a, b = list(map(int, input().strip().split()))
    A.append(a)
    B.append(b)

def put_flower(i, k, n):
    # 如果n大于i，则return
    if n == i:
        return 1
    if n > i:
        return 0
    return put_flower(i, k, n + k) + put_flower(i, k, n + 1)

for j in range(t):
    a, b = A[j], B[j]
    if k == 0:
        print(b-a+1)
    else:
        count = 0
        for i in range(a, b+1):
            n = 0# 从第0个位置开始计算
            count = count + put_flower(i, k, n)
        print(count)