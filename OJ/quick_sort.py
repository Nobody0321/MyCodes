def quick_sort(numbers,start,end):
    if start>=end:
        return
    base = numbers[start]
    i = start
    j = end
    while i != j:
        while numbers[j] >= base and j > i:
            j -= 1
        while numbers[j] <= base and j > i:
            i += 1
        if i < j:
            numbers[i], numbers[j] = numbers[j], numbers[i]
    numbers[start], numbers[j] = numbers[j], numbers[start]

    quick_sort(numbers, start, j-1)
    quick_sort(numbers, j+1, end)


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


if __name__ == "__main__":
    nums =[2,0,1]
    quick_sort(nums,0,len(nums)-1)
    print(quick_sort_not_in_place(nums))