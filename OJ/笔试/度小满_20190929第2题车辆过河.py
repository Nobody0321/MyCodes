# 题目描述：
# 有 N 辆车要陆续通过一座最大承重为 W 的桥，其中第 i 辆车的重量为 w[i]，通过桥的时间为 t[i]。要求： 第 i 辆车上桥的时间不早于第 i - 1 辆车上桥的时间；

# 任意时刻桥上所有车辆的总重量不超过 W。

# 那么，所有车辆都通过这座桥所需的最短时间是多少？

# 输入
# 第一行输入两个整数 N、W（1 <= N、W <= 100000）。第二行输入 N 个整数 w[1] 到 w[N]（1 <= w[i] <= W）。第三行输入 N 个整数 t[1] 到 t[N]（1 <= t[i] <= 10000）。

# 输出
# 输出一个整数，表示所有车辆过桥所需的最短时间。


# 样例输入
# 4 2
# 1 1 1 1
# 2 1 2 2
# 样例输出
# 4

# 提示
# 样例解释
# 不妨设第 1 辆车在 0 时刻上桥，则：
# 第 2 辆车也可以在 0 时刻上桥；
# 第 2 辆车在 1 时刻下桥，此时第 3 辆车上桥；
# 第 1 辆车在 2 时刻下桥，此时第 4 辆车上桥；
# 第 3 辆车在 3 时刻下桥；
# 第 4 辆车在 4 时刻下桥，此时所有车辆都通过这座桥。

def pass_bridge(n, max_w, weights, durations):
    # total_duration[i][j] 记录了[i, j]货车同时出发，过桥的最大通过时间
    total_durations = [[0] * n] * n
    total_durations[0][0] = durations[0]
    dp = [0] * n
    dp[0] = durations[0]

    for i in range(0, n):
        for j in range(i + 1, n):
            if i == j:
                total_durations[i][j] = durations[i]
                continue
            total_durations[i][j] = max(durations[j], total_durations[i][j-1])

    for i in range(1, n):
        # 前面的车都下桥后，第i个车单独过桥的情况下，总用时
        dp[i] += durations[i] + dp[i - 1]
        for j in range(i-1, -1, -1):
            if sum(weights[j:i+1]) <= max_w:
                dp[i] = min(dp[i], dp[j - 1] + total_durations[j][i])
            else:
                break
    
    return dp[-1]

if __name__ == "__main__":
    # n, max_w = list(map(int,input().split()))
    # weights = list(map(int,input().split()))
    # durations = list(map(int,input().split()))
    n, max_w = 4, 2
    weights = [1, 1, 1, 1]
    durations = [2, 1, 2, 2]
    print(pass_bridge(n, max_w, weights, durations))