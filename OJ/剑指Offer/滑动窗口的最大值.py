# class Solution:
#     def maxInWindows(self, num, size):
#         sums = []
#         if size <= 0:
#             return []
#         l = len(num)
#         for i in range(l-size+1):
#             s = max(num[i:i+size])
#             sums.append(s)
#         return sums


# 解法2，使用一个队列动态保存每个窗口最大值的索引
class Solution:
    def maxInWindows(self, num, size):
        queue, ret, i = [], [], 0
        while size and i<len(num):
            # i 是窗口头部的元素索引
            if len(queue) and i + size +1 > queue[0]:
                # 队列头部的元素已过期，不属于当前窗口
                # 保证了queue内只会保存当前窗口内的索引
                queue.pop(0)
            while len(queue) and num[queue[-1]] < num[i]:
                # 窗口内新的元素i大于当前队列尾端的元素
                # 保证queue只保存了q[-1]以后最大元素的索引，且保存在了queue[0]
                queue.pop()
            queue.append(i)
            if i>= size-1:
                # 过了第一个窗口开始计数
                ret.append(num[queue[0]])
            i += 1
        return ret

if __name__ == "__main__":
    print(Solution().maxInWindows([2,3,4,2,6,2,5,1],3))