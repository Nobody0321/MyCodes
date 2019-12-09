n, m = map(int, input().split())

result = 0
door = [0] * 1002
for _ in range(n):
    a, b = map(int, input().split())
    a, b = min(a, b), max(a, b)
    for i in range(a, b+1):
        door[i] = 1

for i in range(m):
    c = int(input())
    if door[c] == 1:
        result += 1

print(result)
