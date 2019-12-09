x,y = list(map(int,(input().split(','))))
 
count = 0

while abs(y-x) > abs(y//2 -x):
    if y%2 ==1:
        count +=1
        y = (y-1)/2
    
count += abs(y-x)
print(int(count))