class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        cur_start = 0
        max_l = 0
        last_seen = {}
        for i, char in enumerate(s):
            if last_seen.get(char, -1) >= cur_start:
                cur_start = last_seen[char] + 1
            max_l = max(i - cur_start + 1, max_l)
            last_seen[char] = i
        return max_l
            
                
if __name__ == "__main__":
    s = "abcabcbb"
    print(Solution().lengthOfLongestSubstring(s))

    s =  "bbbbb"
    print(Solution().lengthOfLongestSubstring(s))

    s = "pwwkew"
    print(Solution().lengthOfLongestSubstring(s))
    
    s = " "
    print(Solution().lengthOfLongestSubstring(s))