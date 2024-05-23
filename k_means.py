import math

# 给出需要分类的点
dots = [{'x': 2, 'y': 10}, {'x': 2, 'y': 5}, {'x': 8, 'y': 4}, {'x': 5, 'y': 8},
        {'x': 7, 'y': 5}, {'x': 6, 'y': 4}, {'x': 1, 'y': 2}, {'x': 4, 'y': 9}]
# 给出初始中心
c_1 = {'x': 2, 'y': 10}
c_2 = {'x': 5, 'y': 8}
c_3 = {'x': 1, 'y': 2}
# 初始化簇
A = []
B = []
C = []
# 开始主循环
while 1:
    # 对每个点进行分类
    for i in dots:
        dist_1 = math.sqrt((i['x'] - c_1['x'])**2 + (i['y'] - c_1['y'])**2)
        dist_2 = math.sqrt((i['x'] - c_2['x'])**2 + (i['y'] - c_2['y'])**2)
        dist_3 = math.sqrt((i['x'] - c_3['x'])**2 + (i['y'] - c_3['y'])**2)
        if dist_1 == min(dist_1, dist_2, dist_3):
            if i not in A:
                A.append(i)
            if i in B:
                B.remove(i)
            if i in C:
                C.remove(i)
        elif dist_2 == min(dist_1, dist_2, dist_3):
            if i not in B:
                B.append(i)
            if i in A:
                A.remove(i)
            if i in C:
                C.remove(i)
        else:
            if i not in C:
                C.append(i)
            if i in A:
                A.remove(i)
            if i in B:
                B.remove(i)
    # 更新中心点的坐标
    c_1_x_sum, c_1_y_sum = 0, 0
    c_2_x_sum, c_2_y_sum = 0, 0
    c_3_x_sum, c_3_y_sum = 0, 0

    for i in A:
        c_1_x_sum += i['x']
        c_1_y_sum += i['y']
    for i in B:
        c_2_x_sum += i['x']
        c_2_y_sum += i['y']
    for i in C:
        c_3_x_sum += i['x']
        c_3_y_sum += i['y']

    c_1_new = {'x': c_1_x_sum / len(A), 'y': c_1_y_sum / len(A)}
    c_2_new = {'x': c_2_x_sum / len(B), 'y': c_2_y_sum / len(B)}
    c_3_new = {'x': c_3_x_sum / len(C), 'y': c_3_y_sum / len(C)}

    if c_1 == c_1_new and c_2 == c_2_new and c_3 == c_3_new:
        break
    c_1, c_2, c_3 = c_1_new, c_2_new, c_3_new

# 打印结果
print(A, B, C)
