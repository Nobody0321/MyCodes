# 最后要得到的是只有一个人的时候（要删除）的数字
# 1. n个人，第一个被删除的数字是(m-1)%n
# 2. 设第二轮开始数字为k,此时整个数组下标的映射关系：
#       k->0, k+1 -> 1, ..., k-3->n-3, k-2->n-2
# 3. 假设第二轮删除的数字是x，
#      由映射关系，n-1删除的数字对应n个数字中的(x+k)%n<=>(x+m)%n
# 4. 第二轮删除的数字是(m-1)%(n-1)
# 5. 假设第三轮开始的数字是o,那么这n-2个数字的映射：
#     o->0, o+1 -> 1, ..., o-2->n-3, o-1->n-2
#     假设最后删除的是y,那么y可以映射成n-1个数字队列中 (y+o)%(n-1)
# 6. 因此要得到最后一个人的时候删除的数字，只需要递归下去就行
class Solution:
    # 递归版
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n <=0 or m <=0:
            return -1
        else:
            return (self.LastRemaining_Solution(n-1,m)+m)%n
    # 非递归
    def LastRemaining_Solution_0(self, n, m):
        if n<=0 or m<=0:
            return -1
        else:
            last = 0
            for i in range(2,n+1):
                last = (last+m)%i
            return last