class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        def str2int(num):
            ret = 0
            l = len(num)
            for i in range(l):
                ret *= 10
                ret += ord(num[i]) - ord('0')
            return ret
        num1, num2 = str2int(num1), str2int(num2)
        print(num1,num2)
        return str(num1*num2)


if __name__ == "__main__":
    print(Solution().multiply("123","456"))