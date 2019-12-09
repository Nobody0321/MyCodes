# 思路：
# 首先找到最少的箱子数，就是找到最大的刚刚好能容纳的topK个箱子
# 然后根据这些箱子的容量和负载，可以计算转移时间
# n = int(input())
# a = list(map(int, input().split(" ")))
# boxes = list(map(int, input().split(" ")))
n = 4
a = [3,3,4,3]
boxes = [4, 7, 6, 5]
d = {(boxes[i],a[i]) for i in range(n)}
total_a = sum(a)
sorted_boxes = sorted(d, key=lambda x: (x[0], x[1]), reverse=True)
for i in range(n):
    if sum(list(map(lambda x: x[0], sorted_boxes[:i+1]))) >= total_a:
        break
t = total_a - sum(list(map(lambda x: x[1], sorted_boxes[:i+1])))
k = i + 1
print(k,t)