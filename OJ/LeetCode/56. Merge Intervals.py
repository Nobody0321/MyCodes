class Solution:
    def merge(self, intervals):
        ret = []
        print(sorted(intervals, key=lambda x:x[0]))
        for each in sorted(intervals, key=lambda x:x[0]):
            print(each)
            if ret and ret[-1][1] >= each[0]:
                ret[-1][1] = max(each[1], ret[-1][1])
            else:
                ret.append(each)
        return ret


if __name__ == "__main__":
    print(Solution().merge([[1,3],[2,6],[8,10],[15,18]]))