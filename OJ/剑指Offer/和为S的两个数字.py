class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # 两个数在数据里距离越远，乘积越小
        # 对于1-n个数，x(n-x)图像在x = n/2最大
        # 使用头尾两个指针，找到相差最远的符合条件的两个数
        i, j = 0,len(array)-1
        while i < j:
            s = array[i] + array[j] 
            if s == tsum:
                return [array[i],array[j]]
            elif s < tsum:
                i += 1
            else:
                j -= 1
        return []

if __name__ == "__main__":
    print(Solution().FindNumbersWithSum([1,2,4,7,11,15],15))