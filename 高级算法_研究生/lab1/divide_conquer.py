import time
import copy
import math
import numpy as np
import matplotlib.pyplot as plt


def graham_scan(point, sort=1):
    sort_p = copy.deepcopy(point)
    if sort == 1:
        sort_p = gs_fc1(sort_p)

    stack = []
    stack.append(sort_p[0])
    stack.append(sort_p[1])
    for i in range(2, len(sort_p)):
        length = len(stack)
        top = stack[length-1]
        next_top = stack[length-2]
        v1 = [sort_p[i][0] - next_top[0], sort_p[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while v1[0]*v2[1] - v1[1]*v2[0] > 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sort_p[i][0] - next_top[0], sort_p[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]
        stack.append(sort_p[i])

    return stack

# 将点按照与p0的极角大小排序
def gs_fc1(point):
    sort_p = []
    sort = sorted(point, key=lambda i: (i[1], i[0]))
    sort_p.append(sort[0])
    temp = []
    for i in range(1, len(sort)):
        y = sort[i][1] - sort[0][1]
        x = sort[i][0] - sort[0][0]
        if x == 0:
            angle = 90
        else:
            s = math.atan(y / x)
            angle = math.degrees(s)
            if angle < 0:
                angle += 180
        temp.append([angle, sort[i]])
    sort_temp = sorted(temp, key=lambda i: (i[0]))
    for i in range(sort_temp.__len__()):
        sort_p.append(sort_temp[i][1])
    return sort_p

def divide_conquer(point):
    p = copy.deepcopy(point)
    # Preprocess
    if len(p) < 3:
        return p
    elif len(p) == 3:
        return gs_fc1(p)
    # Divide
    Q_l, Q_r = [], []
    x_mid = np.median([i[0] for i in p])
    for i in range(len(p)):
        if p[i][0] <= x_mid:
            Q_l.append(p[i])
        else:
            Q_r.append(p[i])
    if len(Q_l) == 0:
        Q_l = [Q_r[-1]]
        Q_r.pop(-1)
    if len(Q_r) == 0:
        Q_r = [Q_l[-1]]
        Q_l.pop(-1)
    # Conquer
    divide_l = divide_conquer(Q_l)
    divide_r = divide_conquer(Q_r)
    # Merge
    if len(divide_l) < 4 or len(divide_r) < 4:
        merge = divide_l + divide_r
        path_point = graham_scan(merge)
    else:
        cnt = len(divide_l) + len(divide_r) - 1
        keypoint = divide_l[0]
        merge = []
        merge.append(divide_l[0])
        top_angle = cal_degree(keypoint, divide_r[0])
        top_ind = 0
        if top_angle > 270:
            top_angle -= 360
        for i in range(1, len(divide_r)):
            angle = cal_degree(keypoint, divide_r[i])
            if angle > 270:
                angle -= 360
            if angle > top_angle:
                top_angle = angle
                top_ind = i
        cnt_l = 1
        cnt_r1 = 0
        cnt_r2 = len(divide_r)-1
        # 类似三路归并排序，排序的key为极角
        while cnt > 0:
            if cnt_l < len(divide_l):
                l_angle = cal_degree(keypoint, divide_l[cnt_l])
                if l_angle > 270:
                    l_angle -= 360
            else:
                l_angle = 360
            if cnt_r1 < top_ind :
                r1_angle = cal_degree(keypoint, divide_r[cnt_r1])
                if r1_angle > 270:
                    r1_angle -= 360
            else:
                r1_angle = 180
            if cnt_r2 >= top_ind :
                r2_angle = cal_degree(keypoint, divide_r[cnt_r2])
                if r2_angle > 270:
                    r2_angle -= 360
            else:
                r2_angle = 180

            if l_angle < r1_angle and l_angle < r2_angle:
                merge.append(divide_l[cnt_l])
                cnt_l += 1
            elif r1_angle < l_angle and r1_angle < r2_angle:
                merge.append(divide_r[cnt_r1])
                cnt_r1 +=1
            elif r2_angle < l_angle and r2_angle < r1_angle:
                merge.append(divide_r[cnt_r2])
                cnt_r2 -=1
            else:
                print()
            cnt -= 1
        path_point = graham_scan(merge, sort=0)
        v1 = [path_point[1][0] -path_point[-1][0], path_point[1][1] - path_point[-1][1]]
        v2 = [path_point[0][0] - path_point[-1][0], path_point[0][1] - path_point[-1][1]]
        if v1[0]*v2[1] - v1[1]*v2[0] > 0:
            path_point.pop(0)

    return path_point

def cal_degree(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    if x == 0:
        angle = 90
    else:
        s = math.atan(y / x)
        angle = math.degrees(s)
        if x < 0:
            angle = angle + 180
        elif y < 0:
            angle += 360
    return angle