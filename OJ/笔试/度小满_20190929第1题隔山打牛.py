# 题目描述：
# “你可曾听闻一招从天而降大掌法？”

#   在一部游戏中有这样一个技能，假设地图是一条直线，长度为n，人物所处的位置是x，则可以对x，2*x和2*x+1三格内的敌人分别造成一点伤害，要求2*x+1<=n。

#   设为这个地图的格子做标记为1-n，第i个格子中有一个血量为a_i的敌人。请问你至少使用多少次技能，可以杀死这个地图上所有敌人。

# 输入
# 输入第一行包含一个正整数n,表示格子的数量(1<=n<=100)

# 输入第二行包含n个正整数a_i,表示第i个格子中敌人的血量。

# 输出
# 输出仅包含一个正整数，即至少使用多少次技能。


# 样例输入
# 5
# 1 2 3 4 5
# 样例输出
# 8
result = 0
def DFS(a, i, n):
    global result
    if 2 * i > n and 2 * i + 1 > n:
        return a[i]
    if 2 * i + 1 > n:
        result = result + a[i]
        a[i] = max(0, a[i] - a[i * 2])
        return a[i]
    else:
        result = result + max(DFS(a, i * 2, n), DFS(a, i * 2 + 1, n))
        a[i] = max(0, a[i] - max(DFS(a, i * 2, n), DFS(a, i * 2 + 1, n)))
        return a[i]
        

if __name__ == "__main__":
    n = int(input())
    enemies = [0]+list(map(int, input().split()))
    ret = DFS(enemies, 1, n) 
    print(result+ret)