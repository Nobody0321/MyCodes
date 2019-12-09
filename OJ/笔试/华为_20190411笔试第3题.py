# 一张NxM的地图上每个点的海拔高度不同  从当前点只能访问上、下、左、右四个点中还没有到
# 达过的点，且下一步选择的点的海拔高度必须高于当前点 求从地图中的点A到点B总的路径条数
# 除以10^9的余数。地图左上角坐标为(0,0).右下角坐标为(N-1,M-1)。
# 输入描述:
# 第一行输入两个整数x,X(0<N<600，0<M<600) 用空格隔开 接下来x行输入，每行X个整数用空格隔开，代表对应位置的海拔高度 (0<海拔高度4360000) 
# 最后一行四个整数X,Y,Z,W 前两个代表3的坐标为(X,Y) 后两个代表B的坐标为(Z, W) 
# 输入保证A、B坐标不同，且坐标合法
# 输出描述:
# 输出一个整数并换行，整数表示从A到B总的路径条数除以10^9的余数
# 输入：
# 4 5
# 0 1 0 0 0
# 0 2 3 0 0
# 0 0 4 5 0
# 0 0 7 6 0
# 0 1 3 2
# 输出：
# 2


# N, M = list(map(int,input().split(' ')))
# Map = []
# for i in range(N):
#     Map.append(input().split(' '))
# X,Y,Z,W =  list(map(int,input().split(' ')))

N, M = 4, 5
Map = [[0,1, 0, 0, 0],
[0, 2, 3, 0, 0],
[0, 0, 4, 5, 0],
[0, 0, 7, 6, 0]]
X,Y,Z,W = 0, 1, 3, 2

def uniquePathsIII(grid,startx,starty,endx, endy):
    si, sj, ei, ej = startx, starty, endx, endy
    visited = [[0] * len(grid[0]) for j in range(len(grid[0]))]
    return count(si, sj, ei, ej, visited, grid)

di = [1, 0, -1, 0]
dj = [0, 1, 0, -1]

def count(si, sj, ei, ej, visited, grid): 
    if si < 0 or si >= len(grid) or sj < 0 or sj >= len(grid[0]):
        # 各坐标取值0~N-1, 0~M-1
        return 0

    if visited[si][sj]:
        # 这个点已经走过，不能再走了
        return 0

    if si == ei and sj == ej:
        # 已到达
        return 1

    else:
        visited[si][sj] = True
        c = 0
        for i in range(len(di)):
            tempi = si + di[i]
            tempj = sj + dj[i]
            if tempi < 0 or tempi >= len(grid) or tempj < 0 or tempj >= len(grid[0]) :
                # 各坐标取值0~N-1,0~M-1
                continue
            else:
                if grid[tempi][tempj] > grid[si][sj] :
                    # 下一个点符合条件，可以走 
                    c += count(tempi, tempj, ei, ej, visited, grid)

        return c

print(uniquePathsIII(Map,X,Y,Z,W))