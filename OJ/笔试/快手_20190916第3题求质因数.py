n = 6
result = 0
for num in range(2, n+1):
    while num > 1:
        for i in range(2, int(num**0.5)+1):
            if num % i == 0:
                result += 1
                num //= i
                break
        else:          #这个表示for循环全跑完执行的，相当于此时num为质数
            result += 1
            break
print(result)

