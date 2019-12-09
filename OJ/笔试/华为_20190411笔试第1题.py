# 连续输入字符里(输入字符里个数为N ,每个字符申长度不大于100 ， 输入字符串间按照空格键分
# 隔)， 请按长度为8拆分每个字符串后输出到新的字符串数组，输出的字符串按照升序排列。
# 长度不是8整数倍的字符时请在后面补数字0，空字符串不处理。
# 输入描述:
# 输入内容: 2 abc -23456789
# 输入说明:
# 1
# 输入两个字符串〈以空格分隔) ，其中一个为abc，另一个为123456789
# 输出描述;
# 输出结果: 12345678 90000000 abc00000
# 输出说明:
# 1
# abc字符串需要再后边补杖，12345789拆分为-2345678与99000000，所有的字符串需要升
# 序排列后输出（以空格分隔) 
def cutBy8(lists):
    temp = []
    for i in range(len(lists)):
        each = lists[i]
        while len(each)>8:
            temp.append(each[:8])
            each = each[8:]
        lists[i] = (each+'00000000')[:8]
    lists.extend(temp)
    lists.sort()
    return lists

inputInfo =  input().split(' ')
N = inputInfo[0]
lists = inputInfo[1:]
print(cutBy8(lists))