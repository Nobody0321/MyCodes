# 智能一点面试题
# 给定一个数组，数组元素之和为1
# 数组中每个元素代表其对应下标出现的概率
# 这就是个概率分布，要求设计一个函数，输出满足该概率分布


def find(k, nums):
    if len(nums) <= 2 :
        return nums[0]
    mid = len(nums)//2
    m = nums[mid]
    if k > m:
        return find(k, nums[mid:])
    elif k < m:
        return find(k, nums[:mid+1])
    else: 
        return m

if __name__ =='__main__':
    import random
    n = [0,0.5,0.2,0.3]
    for i in range(1,len(n)):
        n[i] += n[i-1]

    count = {0:0,1:0,2:0}
    for i in range(10000):
        k = random.random()
        print(n, k)

        target = find(k, n)
        res = n.index(target)
        count[res] += 1
        print(res)

    print(count)