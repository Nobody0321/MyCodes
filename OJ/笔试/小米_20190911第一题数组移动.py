n = int(input())
matric = []
for _ in range(n):
    matric.append(list(map(int, input().split(' '))))
ret = []

for line in matric:
    length = len(line)
    i = 0
    new_line = []
    while i <= length - 1:
        if line[i] == 0:
            i += 1

        # 一个数跟后边一个数相同就可以合并
        elif i < length - 1 and line[i] == line[i + 1]:
            new_line.append(line[i] * 2)
            # 下一个也不用看了
            i += 2
        else:
            # 没法合并
            new_line.append(line[i])
            i += 1
    # 后边补0
    new_line.extend([0] * (length - len(new_line)))
    # 把这一行添加到结果里面
    ret.append(new_line)

# 格式化输出
for line in ret:
    line = ' '.join(list(map(str, line)))
    print(line)
