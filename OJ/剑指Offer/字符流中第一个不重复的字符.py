class Solution:
    # 返回对应char
    def __init__(self):
        # 按顺序维护当前所有的不重复字符
        self.strs = []

    def FirstAppearingOnce(self):
        # 输出当前第一个不重复字符
        return self.strs[0] if len(self.strs) else '#'

    def Insert(self, char):
        if char in self.strs:
            # 重复就删除
            del self.strs[self.strs.index(char)]
        else:
            # 不重复就加入
            self.strs.append(char)