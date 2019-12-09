class Solution:
    def InversePairs0(self, data):
        # 这个方法需要数组中的元素不重复，且效率较低最后超时
        count = 0
        s = sorted(data)
        for i in range(len(data)):
            count += s.index(data[i])
            del s[s.index(data[i])]
        return count

    def InversePairs(self, data):
        l = len(data) - 1
        return self.merge_sort(data, 0, l) % 1000000007

    def merge_sort(self, nums, start, end):
        if end - start < 1:
            return 0
        mid = (end+start) >> 1
        lCount = self.merge_sort(nums, start, mid)
        rCount = self.merge_sort(nums, mid+1, end)
        new_nums = []
        i, j = start, mid+1
        count = 0
        while i <= mid and j <= end:
            if nums[i] > nums[j]:
                new_nums.append(nums[j])
                count += mid - i + 1 
                j += 1
            else:
                new_nums.append(nums[i])
                i += 1
        if i <= mid:
            new_nums+= nums[i:]
        if j <= end:
            new_nums+= nums[j:]
        for k in range(start, end+1):
            nums[k] = new_nums[k-start]
        return lCount + rCount + count


if __name__ == "__main__":
    nums = [7,4,5,3,1,2]
    print(Solution().InversePairs(nums))