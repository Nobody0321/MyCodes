class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        x = str(x)
        return  x == x[::-1]

            
if __name__ == '__main__':
    s = Solution()
    while True: 
        try: 
            number = input('随便输：')
            print(s.isPalindrome(int(number)))
        except: 
            break 