class Solution:
    def __init__(self):
        self.stack = []

    def push(self, node):
        # write code here
        if self.stack == []:
            self.stack.append((node, node))
        else:
           m = self.stack[-1][-1]
           m = m if m < node else node
           self.stack.append((node, m))

    def pop(self):
        ret = self.stack.pop()[0]
        return ret

    def top(self):
        return self.stack[-1][0]

    def min(self):
        return self.stack[-1][-1]