class Solution:
    def reOrderArray0(self, array):
        # write code here
        even, odd = [], []
        for each in array:
            odd.append(each) if each % 2 ==1 else even.append(each)
        return odd + even

    def reOrderArray1(self, array):
        """
        归并排序
        """
        l = len(array)
        if l <= 1:
            return array
        mid = l>> 1
        left = self.reOrderArray(array[:mid])
        right = self.reOrderArray(array[mid:])
        even, odd = [], []
        for each in left:
            if each % 2 == 0:
                even.append(each)
            else:
                odd.append(each)
        for each in right:
            if each % 2 == 0:
                even.append(each)
            else:
                odd.append(each)
        return odd + even

    
    def reOrderArray(self, array):
        l = len(array)
        for _ in range(1,l):
            for j in range(1,l):
                if array[j]%2 ==1 and array[j-1]%2 == 0:
                    array[j-1], array[j] = array[j], array[j-1]
        return array


if __name__ == "__main__":
    print(Solution().reOrderArray([1,2,3,4,5,6,7]))