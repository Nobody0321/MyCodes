class Solution:
    def rectCover(self, number):
 
        res = [0,1,2]
        while len(res) <= number:
            res.append(res[-1] + res[-2])
        return res[number]