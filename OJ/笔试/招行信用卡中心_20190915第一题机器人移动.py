# s = input()
s = "RRLRL"
ret = [0] * len(s)
count = 0
for i in range(len(s)):
    if s[i] == "R":
        # 坡上
        count += 1
    if s[i] == 'L':
        # 不在坡上
        ret[i-1] = count
        count = 0
     
count = 0
for i in range(len(s)-1, -1, -1):
    if s[i] == "L":
        count += 1
    if s[i] == 'R':
        # 不在坡上
        ret[i+1] = count
        count = 0        
for i in range(len(s)-1):
    if ret[i] == 0:
        continue
    
    if ret[i] + ret[i+1] % 2 == 0:
        ret[i] = ret[i+1] = int((ret[i] + ret[i+1]) / 2)
    else:
        if ret[i] % 2 == 0:
            ret[i]= int((ret[i]+ret[i+1]) // 2)
            ret[i+1] = ret[i]+ 1 
        else:
            ret[i+1]= int((ret[i]+ret[i+1]) // 2)
            ret[i] = ret[i+1]+1

print(ret)