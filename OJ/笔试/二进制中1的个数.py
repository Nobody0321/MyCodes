num = int(input().strip())
if num < 2:
    print(num)
count = 0
while num != 0:
    num = num & (num - 1)
    count += 1
print(count)
