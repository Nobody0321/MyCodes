import sys
if __name__ == "__main__":
    # s1 = list(sys.stdin.readline().strip())
    # s2 = list(sys.stdin.readline().strip())
    s1 = 'a#b'
    s2 = 'aCb'
    def do(s1, s2):
        i, j = 0, 0
        l1, l2 = len(s1), len(s2)
        while i < l1 and j < l2:
            if s1[i] == s2[j]:
                i += 1
                j += 1
                continue
            else:
                if s1[i] == '?':
                    # s1[i] = '?' = s2[j]
                    i += 1
                    j += 1
                    continue
                elif s1[i] == '#':
                    if j != l2 -1 and i != l1 -1:
                        # 不是最後一位還要遞歸判斷
                        # 匹配0或1次，遞歸
                        return do(s2[j] + s1[i+1:], s2[j:]) or do(s1[i+1:], s2[j:])
                    else:
                        # 已經到達最後一位， ‘#’ 匹配一位就行
                        return True
                elif s1[i] == '*':
                    if j != l2 -1 and i != l1 -1:
                        while s2[j+1] == s2[j]:
                            j += 1
                        if s2[j+1] == s1[j+1]:
                            i += 1
                            j += 1
                            continue
                        else:
                            return False
        return True

    ret = do(s1, s2)
    if ret :
        print(1)
    else:
        print(0)

