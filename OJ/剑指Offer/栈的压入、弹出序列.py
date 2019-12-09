class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        s = []
        i = 0
        for each in pushV:
            s.append(each)
            while s and popV[i] == s[-1]:
                s.pop()
                i += 1
        return False if s else True


if __name__ == "__main__":
    print(Solution().IsPopOrder([1,2,3,4,5],[4,3,5,1,2,]))