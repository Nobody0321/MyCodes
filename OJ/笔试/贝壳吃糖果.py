# 贝壳 吃糖果
# 只有相同次幂可以合并成新的指数
# 对1 2 2 3 4 4 5 7
#    dic = {1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 7: 1}, k = 0
#     1. 1不能进位， dic.pop(1), dic = {2: 2, 3: 1, 4: 2, 5: 1, 7: 1} K = 1
#     2. 2可以进位 dic = { 3: 2, 4: 2, 5: 1, 7: 1}
#                 dic = { 4: 3, 5: 1, 7: 1}
#                 dic = {4: 1, 5: 2, 7: 1}
#                 dic = {4: 1, 6: 1, 7: 1} 无法进位
#                 dic = {4: 1, 7: 1} 
#                 k = 2
#     3. 两个无法进, K = 4

# parse input
inputs = []
while True:
    s = input()
    if s.strip() == "":
        break
    inputs.append(s)

n = int(inputs[0].strip())
ids = list(map(int, inputs[1].strip().split(' ')))


dic = {}
for each in ids:
    if each in dic.keys():
        dic[each] += 1
    else:
        dic[each] = 1

k= 0
i = 0
while True:
    if not dic.keys():
        break

    if i not in dic.keys():
        i += 1
        continue

    v = dic[i]

    if v == 1:
        k += 1
        dic.pop(i)
    else:
        # v >= 2
        carry = v // 2
        v = v % 2  # v = 1 or 0
        if v: # v = 1
            k += 1
        dic.pop(i)
        if i + 1 not in dic.keys():
            dic[i+1] = carry
        else:
            dic[i+1] += carry  
        
          
return k
