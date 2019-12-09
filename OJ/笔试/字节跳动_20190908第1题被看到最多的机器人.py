n = int(input())
heights = list(map(int, input().split()))


max_robots = 0  # 记录被看到的最多机器人数
cur_robots = 0  # 记录当前能看到的机器人数

max_i = 0  # 记录看到最多的机器人
cur_view_robot_idx = 0  # 记录当前被看到的机器人坐标

for k in range(1, n):
    
    if heights[k] <= heights[cur_view_robot_idx]:
        # 说明有可能看到

        if k - cur_view_robot_idx > 1:
            if heights[k] < heights[k-1]:
                # 隔着别人 而且比你高，只能看到前边的这一个
                # 不必更新当前能看到的机器人数

                # 判断是否更新最大值
                if cur_robots > max_robots:
                    max_robots = cur_robots
                    max_i = cur_view_robot_idx
                
                # 更新当前能被看到的机器人坐标 和能看到的机器人数
                cur_view_robot_idx = k - 1
                cur_robots = 1

            elif heights[k] == heights[k-1]:
                # 隔着别人，而且一样高
                # 判断是否更新最大值
                if cur_robots > max_robots:
                    max_robots = cur_robots
                    max_i = cur_view_robot_idx
                
                # 更新当前能被看到的机器人坐标 和能看到的机器人数
                cur_view_robot_idx = k
                cur_robots = 0
            else:
                # 前边的挡不住
                # 更新当前能看到的机器人数
                cur_robots += 1
                # 判断是否更新最大值
                if cur_robots > max_robots:
                    max_robots = cur_robots
                    max_i = cur_view_robot_idx

                if heights[k] == heights[cur_view_robot_idx]:
                    # 挡住后边

                    # 更新当前能被看到的机器人坐标 和能看到的机器人数
                    cur_view_robot_idx = k
                    cur_robots = 0
        else: 
            # 当前与最大相邻
            # 更新当前能看到的机器人数
            cur_robots += 1
            # 判断是否更新最大值
            if cur_robots > max_robots:
                max_robots = cur_robots
                max_i = cur_view_robot_idx

            if heights[k] == heights[cur_view_robot_idx]:
                # 挡住后边

                # 更新当前能被看到的机器人坐标 和能看到的机器人数
                cur_view_robot_idx = k
                cur_robots = 0


    else:
        # 当前比最大更高 不可能看到，

        # 不必更新当前能看到的机器人数

        # 更新当前能被看到的机器人坐标
        cur_view_robot_idx = k

        # 判断是否更新最大值
        if cur_robots > max_robots:
            max_robots = cur_robots
            max_i = cur_view_robot_idx

        cur_robots = 0

print(heights[max_i])
