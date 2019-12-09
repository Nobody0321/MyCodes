N = int(input())
ans = 1
for i in range(1,N+1):
    if ans % 10 == 0:
        ans = ans / 10
    ans *= i
print(ans%10)