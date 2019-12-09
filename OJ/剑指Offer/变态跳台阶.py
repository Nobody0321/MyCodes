# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        ret = [1,1]
        if number <=1:
            return ret[number]
        while number > 1:
            ret.append(2*ret[-1])
            number -= 1
        return ret[-1]


if __name__ == "__main__":
    print(Solution().jumpFloorII(2)) 