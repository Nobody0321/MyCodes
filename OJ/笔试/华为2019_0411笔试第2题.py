# 给定一个字符串 ,字符串包含数字、大小写字母以及括号 (包括大揪号、中括号和小括号 ) ，括
# 号可以谋春，即括号里面可以出现数字和括号。
# 按照如下的规则对字符串进行展开 ,不需要考虑括号成对不匹配问题,用例保证括号匹配 ,同时
# 用例保证每个数字后面都有括号，不用考虑数字后面没有括号的这种情况，即2a2{b)这种情况不
# 用考虑。
# 1 ) 数字表示括号里的字符串重复的次数，展开后的字符串不包含括号。
# 2 ) 将字符串进行逆序展开。
# 输出最终展开的字符串。
# 示例
# 输入：
# abc3(A)
# 输出：
# AAACBA
s = input()
# s = 'a3(b4{c})' # 测试
# s = '2(a)'
def isNum(c):
    # 判断一个字符串是不是数字字符
    return True if ord('9') >= ord(c) >= ord('0') else False
       
def isLP(c):
    # 判断一个字符串是不是左括号字符
    return True if c == '(' or c == '{' or c == '[' else False

def isRP(c):
    # 判断一个字符串是不是右括号字符
    return True if c == ')' or c == '}' or c == ']' else False

def c2n(c):
    # 数字字符转数字
    return ord(c) - ord('0')

num = ''
nums = []
chars = []

for c in list(s):
    if isNum(c):
        # 数字可能由多个字符组成
        num = num + c

    elif isLP(c):
        # 左括号出现，数字结束，数字字符串转int，入栈
        num = int(num) # str to int
        nums.append(num)
        num = ''
        # 左括号入栈
        chars.append(c)

    elif isRP(c):
        # 找到右括号，说明一对括号结束了，开始计算
        n = nums.pop() # 数字出栈
        ch= ''
        while not isLP(chars[-1]):
            ch = chars.pop() + ch
        chars.pop() # 左括号出
        chars.append(ch*n) # 新字符串进

    else:
        # 不是数字，那就加到chars里面
        chars.append(c)

print(''.join(chars)[::-1]) # 将所有子串连接输出