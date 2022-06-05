class Solution:
    def threeSum_v1(self, nums):
        """所有test case 都能跑通，但是超时了，没有想到较好的优化办法"""
        results = {}
        sums = []
        # calculate all 2 sums, store index and sum result for each
        for i in range(len(nums)-1):
            for j in range(i + 1, len(nums)):
                sums.append([i, j, nums[i] + nums[j]])
        
        # calculate all 3 sums
        for k in range(len(nums)):
            for l in range(len(sums)):
                if k not in sums[l][:2] and nums[k] + sums[l][2] == 0:
                    res = [nums[sums[l][0]], nums[sums[l][1]],nums[k]]
                    res.sort()
                    res_str = ",".join(list(map(lambda x: str(x), res)))
                    results[res_str] = 1
        return list(map(lambda y: [int(each) for each in y.split(",")], results.keys()))

    def threeSum (self, nums):
        """

        """
        nums.sort()
        print(nums)
        results = []
        # divide into n * 2sum
        for i in range(len(nums)):
            # skip same i
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j = i + 1
            k = len(nums) - 1
            if i == 3:
                print(j, k)
            while j < k:
                res = nums[i] + nums[j] + nums[k]

                # if 的每一个分支都需要操作j/k，不然无限循环
                # 每次在两端的index 变动后需要判断是否重复，否则会跳过一些答案，比如当前跟下一个一样，就会跳过[-1,-1,2]
                if res == 0:
                    results.append([nums[i], nums[j], nums[k]])
                    print(i, j, k, [nums[i], nums[j], nums[k]])
    
                    k -= 1
                    j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                elif res > 0:
                    k -= 1

                    while j < k - 1 and nums[k] == nums[k + 1]:
                        k -= 1
                    continue
                elif res < 0:
                    j += 1
                # skip same results
                # print(j, k)
                
                while j < k - 1 and nums[j] == nums[j - 1]:
                    j += 1
        return results

if __name__ == "__main__":
    import time
    start_time = time.time()
    # nums = [-6,14,-11,7,-5,-8,12,-13,-3,-14,7,0,-7,-15,-5,-9,-13,-7,-5,9,8,-13,-6,-8,-12,7,-10,11,8,-14,12,9,-15,-14,1,-5,-7,-10,-10,5,-9,12,12,-1,12,14,-2,-15,-8,0,9,7,2,10,14,-3,2,11,-6,-13,12,13,11,5,14,-11,7,14,-6,12,-4,-7,9,-7,-1,-1,-8,4,-9,-9,-11,-15,5,6,10,4,11,-10,-8,12,-8,-10,10,11,2,9,-15,-14,0,-13,14,11,-5,0,-11,1,6,-12]
    nums = [-2,0,0,2,2]
    print(Solution().threeSum(nums))
    end_time = time.time()
    print(f"{end_time - start_time} s")