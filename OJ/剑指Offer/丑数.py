class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index<7:
            return index
        ret = [1]
        a,b,c = 0,0,0
        count = 1
        while count < index:
            num = min(ret[a]*2, ret[b]*3, ret[c]*5)
            # 最小数等于哪个，就说明要append哪个，append以后，当前指针可以后移，指向加入的数
            if num == ret[a] * 2:
                a += 1
            if num == ret[b] * 3:
                b += 1
            if num == ret[c] * 5:
                c += 1
            ret.append(num)
            count += 1
        return ret[-1]

if __name__ == "__main__":
    print(Solution().GetUglyNumber_Solution(1))