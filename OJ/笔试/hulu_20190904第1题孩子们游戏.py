# n,m = list(map(int, input().split()))
# a = list(map(int, input().split()))
# w = list(map(int, input().split()))
n, m, a, w = 3, 2, [0,0,1], [2,1,1]
ret = 0

def circle(n, m):    
    if n<=0 or m<=0:
        return -1
    else:
        last = 0
        for j in range(2, n+1):
            last = (last+m) % j
        return last
    
for i in range(len(a)):
    if a[i] == 1:
        ret += w[(circle(n, m)-m) % n]
    
print('%.5f' % float(ret/sum(w)))