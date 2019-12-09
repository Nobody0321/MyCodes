class Solution:
    def minNumberInRotateArray(self, rotateArray):
        length = len(rotateArray)
        if length == 0:
           return 0
        elif length == 1:
            return rotateArray[0]
        else:
            left = 0
            right = length - 1
            while left < right:
                mid = (left + right)//2
                if rotateArray[mid] < rotateArray[right]:
                    right = mid
                else:
                    left = mid+1
            return rotateArray[left]