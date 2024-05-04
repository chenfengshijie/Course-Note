import itertools


def is_point_inside_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(p, a, b) < 0.0
    b2 = sign(p, b, c) < 0.0
    b3 = sign(p, c, a) < 0.0

    return (b1 == b2) and (b2 == b3)


def find_convex_hull_brute_force(points):
    potential_hull_points = set(points)

    for a in points:
        for b, c, d in itertools.combinations(points, 3):
            if a in {b, c, d}:
                continue
            if is_point_inside_triangle(a, b, c, d):
                potential_hull_points.discard(a)
                break

    return list(potential_hull_points)


def bf_fc1(point, a, b, c):
    if point[b][0] == point[a][0]:
        return point[c][0] - point[a][0]
    return (
        point[c][1]
        - point[a][1]
        - (
            (point[c][0] - point[a][0])
            * (point[b][1] - point[a][1])
            / (point[b][0] - point[a][0])
        )
    )


def bf_fc2(point, a, b, c, d):
    if all(
        bf_fc1(point, x, y, z) * bf_fc1(point, x, y, w) >= 0
        for x, y, z, w in [(a, b, c, d), (a, c, d, b), (b, c, d, a)]
    ):
        return d
    if all(
        bf_fc1(point, x, y, z) * bf_fc1(point, x, y, w) >= 0
        for x, y, z, w in [(a, b, d, c), (a, d, c, b), (b, d, c, a)]
    ):
        return c
    if all(
        bf_fc1(point, x, y, z) * bf_fc1(point, x, y, w) >= 0
        for x, y, z, w in [(a, c, b, d), (a, d, b, c), (c, d, b, a)]
    ):
        return b
    if all(
        bf_fc1(point, x, y, z) * bf_fc1(point, x, y, w) >= 0
        for x, y, z, w in [(b, c, a, d), (b, d, a, c), (c, d, a, b)]
    ):
        return a
    return -1


def brute_force(point):
    p = point[:]
    a, b, c, d = 0, 1, 2, 3
    delete = -1
    while a < len(p) - 3:
        b, c, d = a + 1, a + 2, a + 3
        while b < len(p) - 2:
            c, d = b + 1, b + 2
            while c < len(p) - 1:
                d = c + 1
                while d < len(p):
                    delete = bf_fc2(p, a, b, c, d)
                    if delete != -1:
                        del p[delete]
                    else:
                        d += 1
                c += 1
            b += 1
        a += 1
    return p


if __name__ == "__main__":
    points = [(1, 1), (4, 4), (1, 4), (4, 1), (3, 2), (2, 2), (5, 4), (7, 6), (9, 2)]
    convex_hull = find_convex_hull_brute_force(points)
    print(convex_hull)
