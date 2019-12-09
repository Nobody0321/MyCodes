class Solution:
    def groupAnagrams(self, strs):
        res ={}
        for i in strs:
            s = ''.join(sorted(i))
            res[s] = res.get(s,[])+[i]
        return list(res.values())
        
if __name__ == "__main__":
    print(Solution().groupAnagrams(['boo','bob']))                  