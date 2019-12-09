# 从第三行开始，除开每一行两端的元素1，每一个元素i都是上一行i与i-1元素之和
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        List = [[1],[1,1]]
        if numRows <=2:
            return List[:numRows]
        
        else:
            while len(List) < numRows:
                List = self.generateNextLevel(List)
        return List

    def generateNextLevel(self, preList,):
        levelNum = len(preList) + 1
        thisLevel = [0] * levelNum
        thisLevel[0],thisLevel[-1] = 1, 1
        preLevel = preList[-1]
        for i in range(1,levelNum -1):
            thisLevel[i] = preLevel[i-1] + preLevel[i]
        preList.append(thisLevel)
        return preList