class Solution:
    def combinationSum2(self, candidates, target):
        candidates = sorted(candidates)
        self.ret = []
        self.dfs(candidates, target, [])
        return self.ret

    def dfs(self, candidates, target, tmp):
        for i, each in enumerate(candidates):
            if each > target:
                # 后面的数字都大于target，没必要再考虑了
                return
            elif each == target:
                tmp = tmp + [each]
                if tmp not in self.ret:
                    self.ret.append(tmp)  
                # 遍历到头
                return
            else:
                # each < target
                tmp.append(each)
                self.dfs(candidates[i+1:], target-each, tmp)
                tmp.pop()


if __name__ == "__main__":
    print(Solution().combinationSum2([10,1,2,7,6,1,5], 8))