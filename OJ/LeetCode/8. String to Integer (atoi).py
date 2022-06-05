class Solution:
    def myAtoi(self, s) -> int:
        if s in ('+', '-',  '.', ""):
            return 0

        for i, c in enumerate(s):
            if c != " ":
                s = s[i:]
                break

        flag = ""
        if s[0] in ("-", "+"):
            flag = s[0]
            s = s[1:]
            if s == "":
                return 
        
        t = 10
        result = 0
        # print(s)
        for c in s:
            try:
                result = int(c) + result * 10
            except:
                break

        result = -result if flag == "-" else result

        if result > 2**31 - 1:
            return 2**31 - 1
        elif result < -2**31:
            return -2**31
        else:
            return result

if __name__ == "__main__":
    print(Solution().myAtoi("   -42"))