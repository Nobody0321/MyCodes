def quicksort(nums):
    '''
    非inplace版快排
    '''
    if len(nums) <= 1:
        return nums

    # 左子数组
    less = []
    # 右子数组
    greater = []
    # 基准数
    base = nums.pop()

    # 对原数组进行划分
    for x in nums:
        if x < base:
            less.append(x)
        else:
            greater.append(x)

    # 递归调用
    return quicksort(less) + [base] + quicksort(greater)


def quicksort_inplace(nums,left,right):
    if left >= right: 
        return 
    i, j = left, right
    base = nums[left]
    while i != j:
        while nums[j] >= base and j > i:
            j -= 1
        while nums[i] <= base and j > i:
            i += 1
        if i < j:
            nums[i], nums[j] = nums[j], nums[i]
        
    nums[left], nums[i] = nums[i], nums[left]

    quicksort_inplace(nums, left, i-1)
    quicksort_inplace(nums, i+1, right)   


def bubble_sort(nums):
    for i in range(len(nums)-1):
        for j in range(len(nums)-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]


def bucket_sort(nums):
    sort = [0] * (len(nums)+1)
    for num in nums:
        sort[num] += 1
    nums = []
    for i in range(len(sort)):
        if sort[i] > 0:
            nums.append(i)
    print(nums)


def Heap_sort(nums):
    pass


if __name__ == "__main__":
    nums = [4,7,5,2,1,3,6]
    # nums = [7,5,6]
    # i = 0
    # j = len(nums) - 1
    quicksort_inplace(nums,0,len(nums)-1)
    # print(bucket_sort(nums))
    print(nums)