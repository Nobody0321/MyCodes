# 归并排序
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


if __name__ == "__main__":
    import random
    nums = [random.randint(1,100) for _ in range(20)]
    print(nums)
    print(merge_sort(nums))