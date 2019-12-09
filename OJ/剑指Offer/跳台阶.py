class Solution:
    def jumpFloor(self, number):
        result = [1,2]
        for i in range(2,number):
            result.append(result[i-1]+result[i-2])
        
        return result[number-1]

if __name__ == "__main__":
    print(Solution().jumpFloor(1))