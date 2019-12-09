class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a = self.B2D(a)
        b = self.B2D(b)
        return self.D2B((a+b))
    
    def B2D(self, binary):
        """
        :type binary: str
        """
        decimal = 0
        lenth = len(binary)
        for i in range(lenth):
            number = int(binary[lenth - i - 1])
            decimal  += 2 ** i * number
        return decimal
    
    def D2B(self, decimal):
        """
        :type decimal: int
        """  
        binary = ''
        if decimal == 1:
            return '1'   
        if decimal == 0:
            return '0'
        while decimal > 1:
            binary = str(decimal % 2) +binary
            decimal =  decimal // 2
        binary = '1' + binary
        return binary


if __name__ == '__main__':
    s = Solution()
    print(s.addBinary('11','0'))