n,k = map(int,input().split())
data = list(map(int,input().split()))
data.sort()
i = 0
min_num = 0
while(i < n and k>0):
    if data[i]>min_num:
        print(data[i]-min_num)
        min_num = data[i]
        k = k-1
    i = i+1
while(k>0):
    print(0)
    k = k-1