# s = "waimai,dache,lvyou,liren,meishi,jiehun,lvyoujingdian,jiaopei,menpiao,jiudian".split(",")
# s = "waimai,lvyou".split(",")
s = input().split(",")
def comp(s1, s2):
    if s1 == "" and s2 != "":
        return False
    elif s2 == "" and s1 != "":
        return True
    elif s1 == s2 == "":
        return "", ""
    if s1[0] == s2[0]:
        return comp(s1[1:], s2[1:])
    else:
        return s1[0] < s2[0]
    return True

for i in range(len(s) - 1):
    for j in range(len(s) - 1, i, -1):
        if comp(s[j-1], s[j]):
            s[j-1], s[j] = s[j], s[j-1]
        
print(",".join(s))