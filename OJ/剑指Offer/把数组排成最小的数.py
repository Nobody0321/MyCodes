class Solution:
    def PrintMinNumber(self, numbers):
        # 根据快排，调整数字顺序
        def quick_sort(numbers, start, end):
            if start >= end:
                return
            base = numbers[start]
            i = start
            j = end
            while i != j:
                while int(numbers[j] + base) > int(base + numbers[j]) and j>i:
                    j -= 1
                while int(numbers[i] + base) < int(base + numbers[i]) and j>i:
                    i += 1
                if i<j:
                    numbers[i], numbers[j] = numbers[j], numbers[i]
            numbers[start], numbers[j] = numbers[j], numbers[start]
            quick_sort(numbers,start,j-1)
            quick_sort(numbers,j+1,end)
        numbers = list(map(str,numbers))
        quick_sort(numbers,0,len(numbers)-1)
        return '0' if numbers[0]=='0' else ''.join(numbers)


if __name__ == "__main__":
    print(Solution().PrintMinNumber([3,32,321]))
