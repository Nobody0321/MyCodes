class Solution:
    # def maximumGap(self, nums: List[int]) -> int:
    #     """
    #     simplest method
    #     """
    #     if len(nums) < 2:
    #         return 0
    #     nums.sort()
    #     m = -1
    #     for i in range(1, len(nums)):
    #         m = max(m, nums[i] - nums[i-1])
    #     return m

    # def maximumGap(self, nums: List[int]) -> int:
    #     if len(nums) < 2:
    #         return 0
    #     nums.sort()
    #     m = []
    #     for i in range(1, len(nums)):
    #         m.append(nums[i] - nums[i-1])
    #     return max(m)

    def maximumGap1(self, nums: List[int]) -> int:
        """use redix sort"""
        radix = 10
        exp = 1
        max_num = max(nums)
        bucket = [[] for i in range(radix)]
        while max_num / exp:
            for num in nums:
                bucket[(num // exp) % radix].append(num)
            nums = []
            for i in range(len(bucket)):
                nums.extend(bucket[i])
                bucket[i] = []
            exp *= radix
        
        max_gap = 0
        for i in range(1, len(nums)):
            max_gap = max(max_gap, nums[i]-nums[i-1])
        return max_gap

    def maximumGap2(self, nums: List[int]) -> int:
        if not nums or len(nums) < 2:
            return 0
        
        max_n = max(nums)
        min_n = min(nums)
        B_size = max(1, (max_n - min_n)//(len(nums)-1))
        B_cnt = (max_n - min_n) // B_size + 1
        B_list = [None] * B_cnt
        for n in nums:
            idx = (n - min_n)//B_size
            if B_list[idx] == None:
                B_list[idx] = (n, n)
            else:
                if n < B_list[idx][0]:
                    B_list[idx] = (n, B_list[idx][1])
                elif n > B_list[idx][1]:
                    B_list[idx] = (B_list[idx][0], n)
        #print(B_list)
        max_gap = 0
        preBucketMax = min_n
        for bucket in B_list:
            if bucket == None:
                continue
            max_gap = max(max_gap, bucket[0] - preBucketMax)
            preBucketMax = bucket[1]
           
        return max_gap
