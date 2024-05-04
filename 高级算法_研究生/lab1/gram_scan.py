import math

def polar_angle(p0, p1=None):
    if p1 == None:
        p1 = anchor
    y_span = p1[1] - p0[1]
    x_span = p1[0] - p0[0]
    
    return math.atan2(y_span, x_span)

def distance(p0, p1=None):
    if p1 == None:
        p1 = anchor
    return (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2

def det(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def graham_scan(points):
    global anchor

    min_idx = None
    for i, (x, y) in enumerate(points):
        if min_idx == None or y > points[min_idx][1] or (y == points[min_idx][1] and x < points[min_idx][0]):
            min_idx = i
    anchor = points[min_idx]
    
    # 如果有多个相同的极角，则只保留距离最远的那个点
    sorted_points = sorted(points, key=lambda p: (polar_angle(p), -distance(p)))

    del sorted_points[0]
    unique_points = [anchor]
    for pt in sorted_points:
        while len(unique_points) > 1 and polar_angle(unique_points[-1], unique_points[-2]) == polar_angle(pt, unique_points[-2]):
            # print("dis",unique_points[-1])
            del unique_points[-1]
        unique_points.append(pt)

    # 构建凸包
    # print(unique_points)
    # print("anchor",anchor)
    hull = [anchor, unique_points[0]]
    for s in unique_points[1:]:
        while len(hull) > 1 and det(hull[-2], hull[-1], s) <= 0:
            x = hull.pop()
            # print(x)
        hull.append(s)

    return hull

if __name__=="__main__":
# 示例
    points = [(1, 1), (4, 4), (1, 4), (4, 1), (3, 2), (2, 2),(5,4),(7,6),(9,2)]
    convex_hull = graham_scan(points)
    print(convex_hull)
    print(polar_angle((1,3),(1,1)))