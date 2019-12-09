import sys
if __name__ == "__main__":
    # 读取第一行的n
    # n = int(input())
    # T = input().strip()
    # m = int(input())
    n, T, m = 9, 'abaabaaba', 3
    s1 = ['aba', 'ab', 'abaaba']
    ret = 0
    for i in range(m):
        s = s1[i]
        # s = input().strip()
        l1 = len(s)
        if n % l1 != 0:
            total_repeat = int(n/l1 + 1)
            repeat = s*total_repeat
            if repeat[:n] == T:
                ret += 1
        elif T[n-l1:] == s:
               ret += 1
        
    print(ret)  