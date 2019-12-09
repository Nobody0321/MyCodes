class Solution:
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = [ c for c in s.upper() if c.isalnum()]
        return s == s[::-1]