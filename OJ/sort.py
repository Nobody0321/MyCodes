def radix_sort(nums, radix=10):
    """a为整数列表， radix为基数"""
    exp = 1
    for i in range(1, K+1): # K次循环
        bucket = [[] for j in range(radix)] # 不能用 [[]]*radix，否则相当于开了radix个完全相同的list对象
        for val in nums:
            bucket[val%(radix**i)//(radix**(i-1))].append(val) # 獲得整數第K位數字 （從低到高）
        del nums[:]
        for each in bucket:
            nums.extend(each) # 桶合并
    return nums


def quick_sort(nums, start, end):
    if start>=end:
        return
    base = nums[start]
    i = start
    j = end
    while i != j:
        while nums[j] >= base and j > i:
            j -= 1
        while nums[j] <= base and j > i:
            i += 1
        if i < j:
            nums[i], nums[j] = nums[j], nums[i]
    nums[start], nums[j] = nums[j], nums[start]

    quick_sort(nums, start, j-1)
    quick_sort(nums, j+1, end)
    return nums


def quick_sort_not_in_place(nums):
    if nums == []:
        return []
    base = nums[0]
    left = []
    right = []
    for i in range(1, len(nums)):
        if nums[i] >= base:
            right.append(nums[i])
        else:
            left.append(nums[i])
    return quick_sort_not_in_place(left) + [base] + quick_sort_not_in_place(right)


def bubble_sort(nums):
    for i in range(len(nums)-1):
        for j in range(len(nums)-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums


def merge_sort(nums):
    l = len(nums)
    if l <= 1:
        return nums
    mid = l >> 1
    left_part, right_part = merge_sort(nums[:mid]), merge_sort(nums[mid:])
    i, j = 0, 0
    new_nums = []
    while i <= len(left_part) - 1 and j <= len(right_part) - 1:
        if left_part[i] <= right_part[j]:
            new_nums.append(left_part[i])
            i += 1
        else:
            new_nums.append(right_part[j])
            j += 1
    while i <= len(left_part) - 1:
        new_nums.append(left_part[i])
        i += 1
    while j <= len(right_part) - 1:
        new_nums.append(right_part[j])
        j += 1
    return new_nums

# 堆排序其实没有真正定义二叉树的结构，但是在数组中使用二叉树的思想进行遍历和排序
# 对所有节点倒序排序，这样保证最大值/最小值一定是从最底层层层上升到最高层
def heap_sort(nums):   
    def adjust_heap(i, l):
        # 从数组中构建堆(升序构建大顶堆，降序构建小顶堆)
        child_idx = i * 2 + 1  # 该节点的左儿子
        while child_idx <= l:
            if increase:
                if child_idx + 1 <= l and nums[child_idx] < nums[child_idx + 1]:
                    # 如果有右儿子结点且更大，切换到右儿子结点
                    child_idx += 1
                if nums[child_idx] > nums[i]:
                    # 子节点中较大的比父节点大，那就交换
                    nums[i], nums[child_idx] = nums[child_idx], nums[i]
                    i = child_idx
                    child_idx = child_idx * 2 + 1
                else:
                    # 后边也不用比了
                    break
            else:
                if child_idx + 1 <= l and nums[child_idx] > nums[child_idx + 1]:
                    # 如果有右儿子结点且更大，切换到右儿子结点
                    child_idx += 1
                if nums[child_idx] < nums[i]:
                    # 子节点中较大的比父节点大，那就交换
                    nums[i], nums[child_idx] = nums[child_idx], nums[i]
                    i = child_idx
                    child_idx = child_idx * 2 + 1
                else:
                    # 后边也不用比了
                    break

    l = len(nums) - 1
    i = l >> 1  # 从最后一个非叶子节点开始（堆是二叉树，叶子最多有(N + 1) / 2）
    # 1. 构建 最大/最小 堆
    while i >= 0:
        adjust_heap(i, l)
        i -= 1
    
    # 2. 取出最大/最小元素，剩下元素重新排序
    while l:
        # 最值 现在一定在堆顶，可以放到数组尾部，然后对剩下的n-1个元素排序
        nums[l], nums[0] = nums[0], nums[l]
        l -= 1
        adjust_heap(0, l)   

    return nums


def insert_sort(nums):
    
    if len(nums) == 1: return nums
    for i in range(1, len(nums)):
        for j in range(i, 0, -1):
            if nums[j] < nums[j-1]: 
                nums[j], nums[j-1] = nums[j-1], nums[j]
            else:
                break
    return nums


def bucket_sort(nums):
    


if __name__ == "__main__":
    import random
    nums = [random.randint(1,100) for _ in range(20)]
    print(nums)