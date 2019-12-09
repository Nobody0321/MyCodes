def decraceAll(nums):
    for i in range(len(nums)):
        if nums[i] > 0:
            nums[i] -= 1

def splitAll(nums):
    ret = []
    for i in range(len(nums)): 
        each = nums[i]
        if each > 1:       
            ret.append(each//2)
            ret.append(each - each//2)
        else:
            ret.append(each)
    return ret

N,k = list(map(int,(input().split(' '))))

count = 0
N = [N]
for i in range(k):
    N = splitAll(N)
    count += 1
# while True:
#     s = set(N)
#     if len(s) == 1 and list(s)[0] == 0:
#         print(count)
#         break
#     decraceAll(N)
#     count += 1
max_n = max(N)
print(count+max_n)