class Solution:
    def GetNumberOfK_0(self, data, k):
        # write code here
        count = 0
        k = str(k)
        for each in data:
            count += str(each).count(k)
        return count

    def GetNumberOfK(self, data, k):
        def GetFirstK(data, k):
            low = 0
            high = len(data) -1 
            while low <= high:
                mid = (low + high) >> 1
                if data[mid] < k:
                    low = mid + 1
                elif data[mid] > k:
                    high = mid - 1
                else:
                    if mid == low or data[mid - 1] != k: 
                        # data[0]或不为k的时候跳出函数
                        return mid
                    else:
                        high = mid - 1
            return -1

        def GetLastK(data, k):
            low = 0
            high = len(data) -1 
            while low <= high:
                mid = (low + high) >> 1
                if data[mid] < k:
                    low = mid + 1
                elif data[mid] > k:
                    high = mid - 1
                else:
                    if mid == high or data[mid + 1] != k: 
                        #data[0]或不为k的时候跳出函数, 当前mid就是要求的位置
                        return mid
                    else:
                        low = mid + 1
            return -1

        if not data:
            return 0
        if GetLastK(data, k) == -1 and GetFirstK(data, k) == -1:
            return 0
        return GetLastK(data, k) - GetFirstK(data, k) + 1