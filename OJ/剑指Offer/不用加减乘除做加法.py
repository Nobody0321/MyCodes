# 两数相加，相当于进位值与非进位值相加
class Solution:
    def Add(self, num1, num2):
        # write code here
        while num2!=0:
            num2, num1 = (num1&num2)<<1, num1^num2 # 非进位部分， 进位部分
        return num1


if __name__ == "__main__":
    print(Solution().Add(5,7))