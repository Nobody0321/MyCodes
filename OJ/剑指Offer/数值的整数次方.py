class Solution:
    def Power(self, base, exponent):
        ret = 1
        flag = exponent > 0 # 有可能有负指数
        exponent = abs(exponent)        
        while exponent > 0:
            if exponent % 2 ==1:
                ret = ret * base
            base *= base # base 用来计算每一位对应的乘子
            exponent >>= 1
        return ret if flag else 1/ret 


if __name__ == "__main__":
    print(Solution().Power(3,6))