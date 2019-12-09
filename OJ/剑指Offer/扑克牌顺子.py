# 1.五张牌有重复（除癞子以外）肯定不行
# 2.五张牌无重复
#   2.1 无赖子：只要最大牌与最小牌相差4
#   2.2 有癞子：只要最大最小牌相差小于4
class Solution:
    def IsContinuous(self, numbers):
        if numbers == []:
            return False
        d = [0]*14
        Min = 14
        Max = -1
        for each in numbers:
            if each == 0:
                continue
            elif d[each] == 1:
                return False
            d[each] += 1
            Min = Min if each > Min else each
            Max = Max if each < Max else each
        return True if Max-Min<=4 else False
            

if __name__ == "__main__":
    print(Solution().IsContinuous([1,3,0,7,0]))