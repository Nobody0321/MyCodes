class Solution:
    def Permutation(self, ss):
        #  可以使用递归，每个字符的全排列，都是这个字符加上其他数的全排列
        def perm(ss):
            if not ss:
                return []
            if len(ss) == 1:
                return ss
                
            ret = []
            for i in range(len(ss)):
                for each in self.Permutation(ss[:i]+ss[i+1:]):
                    ret.append(ss[i]+each)
            return ret
            
        return sorted(set(perm(list(ss))))


if __name__ == "__main__":
    print(Solution().Permutation('aab'))