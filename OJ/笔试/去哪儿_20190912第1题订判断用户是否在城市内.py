# 题目描述： 酒店预定是去哪少U网提供的一项重要服务。
# 当用户打开去哪儿旅行客户端，去哪儿网默认会根据当前用户的经纬度信息，推荐当前城市的酒店。
# 请你设计一个方案来判断用户是否在城市内。 
# 假设： 
# 1、城市的形状都是封闭的多边形。 
# 2、城市中不存在飞地。 
# 3、为了简化计算，坐标点简化成平面坐标，可按平面距离进行计算
# 要求： 输入用户坐标和城市边界坐标，判断用户是否在城市内（用户在边界上算在城市 内）。 

# 输入 

# 输入数据包括多组，每组为一行，
# 对于每一行： 输入N (N>4)个坐标信息，每个坐标由“经度，纬度组成，多个坐标之间用空格隔开。
# 第一个坐标为用户位置坐标，之后的N-I个坐标为按顺序的城市边界坐标数组。 
# 备注：N值不给出，请自己按照空格进行分割，坐标值有可能会带有1位小数。 
# 对于每行输入，输出一行，值为用户是否在城市内（true:在城市内．false：不在城市内)。 

# 样例输入
# 1,1.5 0,0 2,0 1,2 0,2
# 2,1 0,0 2,0 1,1 1,2 0,2

# 样例输出
# true
# false

def is_inside_polygon(pt, poly):
        """
        使用射线法判断点是否在多边形区域内
        从该点向上发出射线，与多边形交点数为奇数，就说明在内部，否则不在
        """
        c = 0
        i = len(poly) - 1
        while i >= 0:
            # i 指向当前点，j指向上一个点，两点确定一条直线
            j = i - 1
            if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or 
            (poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
                # 如果一个点的x坐标在两点之间
                if (pt[1] <= (pt[0] - poly[i][0]) * (poly[j][1] - poly[i][1]) / (
                    poly[j][0] - poly[i][0]) + poly[i][1]):
                    # 并且这个点的y坐标在两点确定的线段下方或者线段上，就说明可以相交
                    c += 1
            i -= 1
        return c % 2 == 1

import sys

while True:
    # 处理输入
    # all_coords = "1,1.5 0,0 2,0 1,2 0,2".split(' ')
    line = sys.stdin.readline().strip()
    if line == '':
        break
    all_coords = line.split(' ')
    user_coords = all_coords[0]
    city_borders = all_coords[1:]
    # "3,5" -> ['3', '5']
    user_coords = user_coords.split(',')
    # ['3', '5'] -> [3, 5]
    user_coords = list(map(float, user_coords))
    # "1,2 3,4" -> [['1','2'], ['3','4']]
    city_borders = list(map(lambda x: x.split(','), city_borders))
    for i in range(len(city_borders)):
        city_borders[i] = [float(city_borders[i][0]), float(city_borders[i][1])]

    print(is_inside_polygon(user_coords, city_borders))