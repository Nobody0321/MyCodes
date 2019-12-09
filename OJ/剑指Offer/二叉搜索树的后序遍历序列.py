class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False
        root = sequence[-1]
        l = len(sequence)-1
        i = 0
        for i in range(l):
            if sequence[i] > root:
                break
        for j in range(l-1):
            if j < i:
                if sequence[j] > root:
                    return False 
            elif j >= i:
                if sequence[j] < root:
                    return False
        if i>= 1:
            self.VerifySquenceOfBST(sequence[:i]) and self.VerifySquenceOfBST(sequence[i:-1])
        return True


if __name__ == "__main__":
    print(Solution().VerifySquenceOfBST([4,7,5,11,13,12,10]))