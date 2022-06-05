class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        o_x = x
        r_x = 0
        while x:
            r_x = r_x * 10 + (x % 10)
            x = x // 10
        return r_x == o_x

if __name__ == "__main__":
    print(Solution().isPalindrome(10))
