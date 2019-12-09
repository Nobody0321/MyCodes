class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        flag = 1 if (dividend >= 0) == (divisor >= 0) else -1
        dd = abs(dividend)
        dr = abs(divisor)
        if dd < dr:
            return 0
        if dd == dr:
            return flag
        if dr == 1:
            dd = dd if flag > 0 else -dd
            return min(2**31-1, dd)

if __name__ == "__main__":
    print(Solution().divide(10,3))