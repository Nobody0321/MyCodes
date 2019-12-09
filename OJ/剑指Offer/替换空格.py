# 题目描述
# 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

# 思路：
# 正向查找替换的话，"%20"是三个字符，替换" "这一个字符就会覆盖后面的两个字符，使用两个字符串的话又会造成额外的空间开销
# 为了尽可能减少开销，我们先计算替换后的总长度，然后倒序替换，这样在替换的过程中也不会覆盖原字符，可以达到in_place的效果

class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        newLen = 0
        spaceCount = 0
        l = 0
        for c in s:
            l += 1
            if c ==' ':
                spaceCount += 1
        newLen = l + 2* spaceCount
        s = s + ' ' * 2*spaceCount
        s = list(s)
        newLen = newLen -1
        l = l-1
        while newLen >= 0 and newLen >= l:
            if s[l] == ' ':
                s[newLen-2], s[newLen-1], s[newLen] = '%20'
                newLen -= 3
            else:
                s[newLen] = s[l]
                newLen -= 1
            l -= 1

        return ''.join(s)


if __name__ == "__main__":
    print(Solution().replaceSpace('We Are Happy.'))