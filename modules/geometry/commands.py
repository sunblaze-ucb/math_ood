import pdb
import numpy as np
import geo_types as gt
import random
from typing import List, Tuple, Union, Optional, Any

def angle_ppp(p1: gt.Point, p2: gt.Point, p3: gt.Point) -> gt.Angle:
    assert(not np.isclose(p1.a, p2.a).all())
    assert(not np.isclose(p2.a, p3.a).all())
    assert(not np.isclose(p3.a, p1.a).all())
    return gt.Angle(p2.a, p2.a-p1.a, p2.a-p3.a)

def angular_bisector_ll(l1: gt.Line, l2: gt.Line) -> List[gt.Line]:
    assert(not np.isclose(l1.n, l2.n).all())
    x = intersect_ll(l1, l2)
    n1, n2 = l1.n, l2.n
    if np.dot(n1, n2) > 0: n = n1 + n2
    else: n = n1 - n2
    return [
        gt.Line(vec, np.dot(vec, x.a))
        for vec in (n, gt.vector_perp_rot(n))
    ]

def angular_bisector_ppp(p1: gt.Point, p2: gt.Point, p3: gt.Point) -> gt.Line:
    assert(not np.isclose(p1.a, p2.a).all())
    assert(not np.isclose(p2.a, p3.a).all())
    assert(not np.isclose(p3.a, p1.a).all())
    v1 = p2.a - p1.a
    v2 = p2.a - p3.a
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    if np.dot(v1, v2) < 0: n = v1-v2
    else: n = gt.vector_perp_rot(v1+v2)
    return gt.Line(n, np.dot(p2.a, n))

def angular_bisector_ss(l1: gt.Segment, l2: gt.Segment) -> List[gt.Line]:
    return angular_bisector_ll(l1, l2)

def are_collinear_ppp(p1: gt.Point, p2: gt.Point, p3: gt.Point) -> gt.Boolean:
    return gt.Boolean(np.linalg.matrix_rank([p1.a-p2.a, p1.a-p3.a]) <= 1)

def are_concurrent_lll(l1: gt.Line, l2: gt.Line, l3: gt.Line) -> gt.Boolean:
    lines = l1,l2,l3
    
    differences = []
    for i in range(3):
        remaining = [l.n for l in lines[:i]+lines[i+1:]]
        prod = np.abs(np.cross(*remaining))
        differences.append((prod, i, lines[i]))

    l1, l2, l3 = tuple(zip(*sorted(differences)))[2]
    x = intersect_ll(l1, l2)
    return gt.Boolean(np.isclose(np.dot(x.a, l3.n), l3.c))

def are_concurrent(o1: Union[gt.Line, gt.Circle], o2: Union[gt.Line, gt.Circle], o3: Union[gt.Line, gt.Circle]) -> gt.Boolean:
    cand = []
    try:
        #if True:
        if isinstance(o1, gt.Line) and isinstance(o2, gt.Line):
            cand = [intersect_ll(o1, o2)]  # Wrap single point in list
        elif isinstance(o1, gt.Line) and isinstance(o2, gt.Circle):
            cand = intersect_lc(o1, o2)
        elif isinstance(o1, gt.Circle) and isinstance(o2, gt.Line):
            cand = intersect_cl(o1, o2)
        elif isinstance(o1, gt.Circle) and isinstance(o2, gt.Circle):
            cand = intersect_cc(o1, o2)
    except: pass

    for p in cand:
        for obj in (o1,o2,o3):
            if not obj.contains(p.a): break
        else: return gt.Boolean(True)

    return gt.Boolean(False)

def are_concyclic_pppp(p1: gt.Point, p2: gt.Point, p3: gt.Point, p4: gt.Point) -> gt.Boolean:
    z1, z2, z3, z4 = (gt.a_to_cpx(p.a) for p in (p1, p2, p3, p4))
    cross_ratio = (z1-z3)*(z2-z4)*(((z1-z4)*(z2-z3)).conjugate())
    return gt.Boolean(np.isclose(cross_ratio.imag, 0))

def are_congruent_aa(a1: gt.Angle, a2: gt.Angle) -> gt.Boolean:
    #print(a1.angle, a2.angle)
    result = np.isclose((a1.angle-a2.angle+1)%(2*np.pi), 1)
    result = (result or np.isclose((a1.angle+a2.angle+1)%(2*np.pi), 1))
    return gt.Boolean(result)

def are_complementary_aa(a1: gt.Angle, a2: gt.Angle) -> gt.Boolean:
    #print(a1.angle, a2.angle)
    result = np.isclose((a1.angle-a2.angle)%(2*np.pi), np.pi)
    result = (result or np.isclose((a1.angle+a2.angle)%(2*np.pi), np.pi))
    return gt.Boolean(result)

def are_congruent_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Boolean:
    l1, l2 = (
        np.linalg.norm(s.end_points[1] - s.end_points[0])
        for s in (s1, s2)
    )
    return gt.Boolean(np.isclose(l1, l2))

def are_equal_mm(m1: Union[gt.Measure, float, int], m2: Union[gt.Measure, float, int]) -> gt.Boolean:
    # Handle both Measure objects and numeric values
    if isinstance(m1, gt.Measure) and isinstance(m2, gt.Measure):
        assert(m1.dim == m2.dim)
        return gt.Boolean(np.isclose(m1.x, m2.x))
    elif isinstance(m1, gt.Measure):
        # m2 is a numeric value
        return gt.Boolean(np.isclose(m1.x, float(m2)))
    elif isinstance(m2, gt.Measure):
        # m1 is a numeric value
        return gt.Boolean(np.isclose(float(m1), m2.x))
    else:
        # Both are numeric values
        return gt.Boolean(np.isclose(float(m1), float(m2)))

def are_equal_mi(m: gt.Measure, i: int) -> gt.Boolean:
    assert(m.dim == 0)
    return gt.Boolean(np.isclose(m.x, i))

def are_equal_pp(p1: gt.Point, p2: gt.Point) -> gt.Boolean:
    return gt.Boolean(np.isclose(p1.a, p2.a).all())

def are_parallel_ll(l1: gt.Line, l2: gt.Line) -> gt.Boolean:
    if np.isclose(l1.n, l2.n).all(): return gt.Boolean(True)
    if np.isclose(l1.n, -l2.n).all(): return gt.Boolean(True)
    return gt.Boolean(False)

def are_parallel_ls(l: gt.Line, s: gt.Segment) -> gt.Boolean:
    return are_parallel_ll(l, s)

def are_parallel_rr(r1: gt.Ray, r2: gt.Ray) -> gt.Boolean:
    return are_parallel_ll(r1, r2)

def are_parallel_sl(s: gt.Segment, l: gt.Line) -> gt.Boolean:
    return are_parallel_ll(s, l)

def are_parallel_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Boolean:
    return are_parallel_ll(s1, s2)

def are_perpendicular_ll(l1: gt.Line, l2: gt.Line) -> gt.Boolean:
    if np.isclose(l1.n, l2.v).all(): return gt.Boolean(True)
    if np.isclose(l1.n, -l2.v).all(): return gt.Boolean(True)
    return gt.Boolean(False)

def are_perpendicular_lr(l: gt.Line, r: gt.Ray) -> gt.Boolean:
    return are_perpendicular_ll(l, r)

def are_perpendicular_rl(r: gt.Ray, l: gt.Line) -> gt.Boolean:
    return are_perpendicular_ll(r, l)

def are_perpendicular_sl(s: gt.Segment, l: gt.Line) -> gt.Boolean:
    return are_perpendicular_ll(s, l)

def are_perpendicular_ls(l: gt.Line, s: gt.Segment) -> gt.Boolean:
    return are_perpendicular_ll(l, s)

def are_perpendicular_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Boolean:
    return are_perpendicular_ll(s1, s2)

def area_P(polygon: gt.Polygon) -> gt.Measure:
    p0 = polygon.points[0].a
    vecs = [p.a - p0 for p in polygon.points[1:]]
    cross_sum = sum(
        np.cross(v1, v2)
        for v1, v2 in zip(vecs, vecs[1:])
    )
    return gt.Measure(abs(cross_sum)/2, 2)

def center_c(c: gt.Circle) -> gt.Point:
    return gt.Point(c.c)

def circle_pp(center: gt.Point, passing_point: gt.Point) -> gt.Circle:
    return gt.Circle(center.a, np.linalg.norm(center.a - passing_point.a))

def circle_ppp(p1: gt.Point, p2: gt.Point, p3: gt.Point) -> gt.Circle:
    axis1 = line_bisector_pp(p1, p2)
    axis2 = line_bisector_pp(p1, p3)
    center = intersect_ll(axis1, axis2)
    return circle_pp(center, p1)

def circle_pm(p: gt.Point, m: Union[gt.Measure, int]) -> gt.Circle:
    if isinstance(m, gt.Measure):
        assert(m.dim == 1)
        return gt.Circle(p.a, m.x)
    else:
        return gt.Circle(p.a, float(m))

def contained_by_pc(point: gt.Point, by_circle: gt.Circle) -> gt.Boolean:
    return gt.Boolean(by_circle.contains(point.a))

def contained_by_pl(point: gt.Point, by_line: gt.Line) -> gt.Boolean:
    return gt.Boolean(by_line.contains(point.a))

def distance_pp(p1: gt.Point, p2: gt.Point) -> gt.Measure:
    return gt.Measure(np.linalg.norm(p1.a-p2.a), 1)

def equality_aa(a1: gt.Angle, a2: gt.Angle) -> gt.Boolean:
    return are_congruent_aa(a1, a2)

def equality_mm(m1: gt.Measure, m2: gt.Measure) -> gt.Boolean:
    assert(m1.dim == m2.dim)
    return gt.Boolean(np.isclose(m1.x, m2.x))

def equality_ms(m: gt.Measure, s: gt.Segment) -> gt.Boolean:
    assert(m.dim == 1)
    return gt.Boolean(np.isclose(m.x, s.length))

def equality_mi(m: gt.Measure, i: int) -> gt.Boolean:
    assert(m.dim == 0 or i == 0)
    return gt.Boolean(np.isclose(m.x, i))

def equality_pp(p1: gt.Point, p2: gt.Point) -> gt.Boolean:
    return are_equal_pp(p1, p2)

def equality_Pm(polygon: gt.Polygon, m: gt.Measure) -> gt.Boolean:
    assert(m.dim == 2)
    return gt.Boolean(np.isclose(area_P(polygon).x, m.x))

def equality_PP(poly1: gt.Polygon, poly2: gt.Polygon) -> gt.Boolean:
    return gt.Boolean(np.isclose(area_P(poly1).x, area_P(poly2).x))

def equality_sm(s: gt.Segment, m: gt.Measure) -> gt.Boolean:
    return equality_ms(m,s)

def equality_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Boolean:
    return gt.Boolean(np.isclose(s1.length, s2.length))

def equality_si(s: gt.Segment, i: int) -> None:
    pass # TODO

def intersect_ll(line1: gt.Line, line2: gt.Line) -> gt.Point:
    matrix = np.stack((line1.n, line2.n))
    b = np.array((line1.c, line2.c))
    assert(not np.isclose(np.linalg.det(matrix), 0))
    return gt.Point(np.linalg.solve(matrix, b))

def intersect_lc(line: gt.Line, circle: gt.Circle) -> List[gt.Point]:
    # shift circle to center
    y = line.c - np.dot(line.n, circle.c)
    x_squared = circle.r_squared - y**2
    if np.isclose(x_squared, 0): 
        return [gt.Point(y*line.n + circle.c)]  # Wrap single point in a list
    assert(x_squared > 0)

    x = np.sqrt(x_squared)
    intersections = [
        gt.Point(x*line.v + y*line.n + circle.c),
        gt.Point(-x*line.v + y*line.n + circle.c),
    ]
    return random.shuffle(intersections)

def intersect_cc(circle1: gt.Circle, circle2: gt.Circle) -> List[gt.Point]:
    center_diff = circle2.c - circle1.c
    center_dist_squared = np.dot(center_diff, center_diff)
    center_dist = np.sqrt(center_dist_squared)
    relative_center = (circle1.r_squared - circle2.r_squared) / center_dist_squared
    center = (circle1.c + circle2.c)/2 + relative_center*center_diff/2

    rad_sum  = circle1.r + circle2.r
    rad_diff = circle1.r - circle2.r
    det = (rad_sum**2 - center_dist_squared) * (center_dist_squared - rad_diff**2)
    if np.isclose(det, 0): 
        return [gt.Point(center)]  # Already returning a list
    assert(det > 0)
    center_deviation = np.sqrt(det)
    center_deviation = np.array(((center_deviation,),(-center_deviation,)))

    intersections = [
        gt.Point(center + center_dev)
        for center_dev in center_deviation * 0.5*gt.vector_perp_rot(center_diff) / center_dist_squared
    ]
    return random.shuffle(intersections)

def intersect_cl(c: gt.Circle, l: gt.Line) -> List[gt.Point]:
    return intersect_lc(l,c)

def intersect_Cl(arc: gt.Arc, line: gt.Line) -> List[gt.Point]:
    results = intersect_lc(line, arc)
    return [x for x in results if arc.contains(x.a)]

def intersect_cs(circle: gt.Circle, segment: gt.Segment) -> List[gt.Point]:
    results = intersect_lc(segment, circle)
    return [x for x in results if segment.contains(x.a)]

def intersect_ls(line: gt.Line, segment: gt.Segment) -> gt.Point:
    result = intersect_ll(line, segment)
    assert(segment.contains(result.a))
    return result

def intersect_sl(segment: gt.Segment, line: gt.Line) -> gt.Point:
    return intersect_ls(line, segment)

def intersect_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Point:
    result = intersect_ll(s1, s2)
    assert(s1.contains(result.a))
    assert(s2.contains(result.a))
    return result

def line_bisector_pp(p1: gt.Point, p2: gt.Point) -> gt.Line:
    p = (p1.a+p2.a)/2
    n = p2.a-p1.a
    assert((n != 0).any())
    return gt.Line(n, np.dot(n,p))

def line_bisector_s(segment: gt.Segment) -> gt.Line:
    p1, p2 = segment.end_points
    p = (p1+p2)/2
    n = p2-p1
    return gt.Line(n, np.dot(n,p))

def line_pl(point: gt.Point, line: gt.Line) -> gt.Line:
    return gt.Line(line.n, np.dot(line.n, point.a))

def line_pp(p1: gt.Point, p2: gt.Point) -> gt.Line:
    assert((p1.a != p2.a).any())
    n = gt.vector_perp_rot(p1.a-p2.a)
    return gt.Line(n, np.dot(p1.a, n))

def line_ps(point: gt.Point, segment: gt.Segment) -> gt.Line:
    return line_pl(point, segment)

def midpoint_pp(p1: gt.Point, p2: gt.Point) -> gt.Point:
    return gt.Point((p1.a+p2.a)/2)

def midpoint_s(segment: gt.Segment) -> gt.Point:
    p1, p2 = segment.end_points
    return gt.Point((p1+p2)/2)

def minus_mm(m1: Union[gt.Measure, float, int], m2: Union[gt.Measure, float, int]) -> gt.Measure:
    # Handle both Measure objects and numeric values
    if isinstance(m1, gt.Measure) and isinstance(m2, gt.Measure):
        assert(m1.dim == m2.dim)
        return gt.Measure(m1.x - m2.x, m1.dim)
    elif isinstance(m1, gt.Measure):
        # m2 is a numeric value
        # Assume m2 has the same dimension as m1
        return gt.Measure(m1.x - float(m2), m1.dim)
    elif isinstance(m2, gt.Measure):
        # m1 is a numeric value
        # Assume m1 has the same dimension as m2
        return gt.Measure(float(m1) - m2.x, m2.dim)
    else:
        # Both are numeric values - assume they are dimensionless
        return gt.Measure(float(m1) - float(m2), 0)

def minus_ms(m: gt.Measure, s: gt.Segment) -> gt.Measure:
    assert(m.dim == 1)
    return gt.Measure(m.x-s.length, 1)

def minus_sm(s: gt.Segment, m: gt.Measure) -> gt.Measure:
    assert(m.dim == 1)
    return gt.Measure(s.length-m.x, 1)

def minus_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Measure:
    return gt.Measure(s1.length-s2.length, 1)

def mirror_cl(circle: gt.Circle, by_line: gt.Line) -> gt.Circle:
    return gt.Circle(
        center = circle.c + by_line.n*2*(by_line.c - np.dot(circle.c, by_line.n)),
        r = circle.r,
    )

def mirror_cp(circle: gt.Circle, by_point: gt.Point) -> gt.Circle:
    return gt.Circle(
        center = 2*by_point.a - circle.c,
        r = circle.r
    )

def mirror_ll(line: gt.Line, by_line: gt.Line) -> gt.Line:
    n = line.n - by_line.n * 2*np.dot(line.n, by_line.n)
    return gt.Line(n, line.c + 2*by_line.c * np.dot(n, by_line.n) )

def mirror_lp(line: gt.Line, by_point: gt.Point) -> gt.Line:
    return gt.Line(line.n, 2*np.dot(by_point.a, line.n) - line.c)

def mirror_pc(point: gt.Point, by_circle: gt.Circle) -> gt.Point:
    v = point.a - by_circle.c
    assert(not np.isclose(v,0).all())
    return gt.Point(by_circle.c + v * (by_circle.r_squared / gt.square_norm(v)) )

def mirror_pl(point: gt.Point, by_line: gt.Line) -> gt.Point:
    assert not np.isclose(np.dot(point.a, by_line.n) - by_line.c, 0), "Point must not be on the line"
    return gt.Point(point.a + by_line.n*2*(by_line.c - np.dot(point.a, by_line.n)))

def mirror_pp(point: gt.Point, by_point: gt.Point) -> gt.Point:
    return gt.Point(2*by_point.a - point.a)

def mirror_ps(point: gt.Point, segment: gt.Segment) -> gt.Point:
    return mirror_pl(point, segment)

def orthogonal_line_pl(point: gt.Point, line: gt.Line) -> gt.Line:
    return gt.Line(line.v, np.dot(line.v, point.a))

def orthogonal_line_ps(point: gt.Point, segment: gt.Segment) -> gt.Line:
    return orthogonal_line_pl(point, segment)

def point_() -> gt.Point:
    return gt.Point(np.random.normal(size = 2))

def point_c(circle: gt.Circle) -> gt.Point:
    return gt.Point(circle.c + circle.r * gt.random_direction())

def point_l(line: gt.Line) -> gt.Point:
    return gt.Point(line.c * line.n + line.v * np.random.normal() )

def point_s(segment: gt.Segment) -> gt.Point:
    return gt.Point(gt.interpolate(segment.end_points[0], segment.end_points[1], np.random.random()))

def point_pm(point: gt.Point, distance: int) -> gt.Point:
    """Create a point at a specified distance from an existing point in a random direction."""
    assert(distance > 0)
    return gt.Point(point.a + distance * gt.random_direction())

def polar_pc(point: gt.Point, circle: gt.Circle) -> gt.Line:
    n = point.a - circle.c
    assert(not np.isclose(n, 0).all())
    return gt.Line(n, np.dot(n, circle.c) + circle.r_squared)
# note: polygon_ppi removed because describing it in natural language was prohibitive.

# note: no polygon command for arbitrary points, since we could not ensure the points would be listed in order
# triangle and quadrilateral could be defined with explicit casing
# but mech translator and classical generator would need to handle the output args specially
# also, triangles should have a lot of special operations, such as incircle, circumcircle, etc.

def power_mi(m: gt.Measure, i: int) -> gt.Measure:
    assert(i == 2)
    return gt.Measure(m.x ** i, m.dim*i)

def power_si(s: gt.Segment, i: int) -> gt.Measure:
    return gt.Measure(s.length ** i, i)

def prove_b(x: gt.Boolean) -> gt.Boolean:
    return x

def radius_c(circle: gt.Circle) -> gt.Measure:
    return gt.Measure(circle.r, 1)

def ratio_mm(m1: gt.Measure, m2: gt.Measure) -> gt.Measure:
    assert(not np.isclose(m1.x, 0))
    return gt.Measure(m1.x / m2.x, m1.dim - m2.dim)


'''
# too much trouble to check
def rotate_pap(point: gt.Point, angle: gt.Angle, by_point: gt.Point) -> gt.Point:
    return gt.Point(by_point.a + gt.rotate_vec(point.a - by_point.a, angle.angle))
'''

def rotate_pAp(point: gt.Point, angle_size: gt.AngleSize, by_point: gt.Point) -> gt.Point:
    return gt.Point(by_point.a + gt.rotate_vec(point.a - by_point.a, angle_size.x))

def segment_pp(p1: gt.Point, p2: gt.Point) -> gt.Segment:
    return gt.Segment(p1.a, p2.a)

def sum_mm(m1: Union[gt.Measure, float, int], m2: Union[gt.Measure, float, int]) -> gt.Measure:
    # Handle both Measure objects and numeric values
    if isinstance(m1, gt.Measure) and isinstance(m2, gt.Measure):
        assert(m1.dim == m2.dim)
        return gt.Measure(m1.x + m2.x, m1.dim)
    elif isinstance(m1, gt.Measure):
        # m2 is a numeric value
        # Assume m2 has the same dimension as m1
        return gt.Measure(m1.x + float(m2), m1.dim)
    elif isinstance(m2, gt.Measure):
        # m1 is a numeric value
        # Assume m1 has the same dimension as m2
        return gt.Measure(float(m1) + m2.x, m2.dim)
    else:
        # Both are numeric values - assume they are dimensionless
        return gt.Measure(float(m1) + float(m2), 0)

def sum_ms(m: gt.Measure, s: gt.Segment) -> gt.Measure:
    assert(m.dim == 1)
    return gt.Measure(m.x + s.length, 1)

def sum_mi(m: gt.Measure, i: int) -> gt.Measure:
    assert(m.dim == 0)
    return gt.Measure(m.x + i, 0)

def sum_ss(s1: gt.Segment, s2: gt.Segment) -> gt.Measure:
    return gt.Measure(s1.length + s2.length, 1)

def tangent_pc(point: gt.Point, circle: gt.Circle) -> List[gt.Line]:
    polar = polar_pc(point, circle)
    intersections = intersect_lc(polar, circle)
    if len(intersections) == 2:
        return [line_pp(point, x) for x in intersections]
    else: 
        return [polar]  # Wrap single line in a list

def touches_cc(c1: gt.Circle, c2: gt.Circle) -> gt.Boolean:
    lens = c1.r, c2.r, np.linalg.norm(c1.c-c2.c)
    return gt.Boolean(np.isclose(sum(lens), 2*max(lens)))

def touches_lc(line: gt.Line, circle: gt.Circle) -> gt.Boolean:
    return gt.Boolean(
        np.isclose(circle.r, np.abs(np.dot(line.n, circle.c) - line.c) )
    )

def touches_cl(circle: gt.Circle, line: gt.Line) -> gt.Boolean:
    return touches_lc(line, circle)

def translate_pv(point: gt.Point, vector: gt.Vector) -> gt.Point:
    return gt.Point(point.a + vector.v)

def vector_pp(p1: gt.Point, p2: gt.Point) -> gt.Vector:
    return gt.Vector((p1.a, p2.a))

def measure(x: Any) -> Any:
    """
    Marks an element as the final measurement for a construction file.
    Simply returns the input element.
    """
    return x


def point_at_distance_along_line(line: gt.Line, reference_point: gt.Point, distance: float) -> gt.Point:
    """Create a point on a line at a specified distance from the closest point on the line to a reference point."""
    # Project reference point onto the line
    closest_pt = line.c * line.n - np.dot(reference_point.a, line.n) * line.n + reference_point.a
    # Move along line direction by the specified distance
    # Note that because the command doesn't naturally specify which direction,
    # we actually don't want the outcome to be deterministic, or else the problem statement
    # will not match the construction's logic.
    if random.random() < 0.5:
        return gt.Point(closest_pt + line.v * distance)
    else:
        return gt.Point(closest_pt - line.v * distance)


def circumcircle_p(p: gt.Polygon) -> gt.Circle:
    return gt.Circle(p.center, np.linalg.norm(p.points[0].a - p.center.a))


def triangle_ppp(p1: gt.Point, p2: gt.Point, p3: gt.Point) -> Tuple[gt.Triangle, gt.Segment, gt.Segment, gt.Segment]:
    triangle = gt.Triangle(p1, p2, p3)
    return triangle, *triangle.segments

def circumcircle_t(t: gt.Triangle) -> gt.Circle:
    return circle_ppp(t.a, t.b, t.c)

def circumcenter_t(t: gt.Triangle) -> gt.Point:
    return centroid_t(t)

def circumradius_t(t: gt.Triangle) -> gt.Measure:
    return radius_c(circumcircle_t(t))

def centroid_t(t: gt.Triangle) -> gt.Point:
    median_a = segment_pp(t.a, midpoint_pp(t.b, t.c))
    median_b = segment_pp(t.b, midpoint_pp(t.a, t.c))
    return intersect_ss(median_a, median_b)

def incircle_t(t: gt.Triangle) -> gt.Circle:
    """Returns the incircle of a triangle - the circle tangent to all three sides."""
    # Get the angle bisectors
    ab1 = angular_bisector_ppp(t.b, t.a, t.c)
    ab2 = angular_bisector_ppp(t.a, t.b, t.c)
    
    # Incenter is the intersection of angle bisectors
    incenter = intersect_ll(ab1, ab2)
    
    # Calculate distance from incenter to any side (all are equal)
    side_a = line_pp(t.b, t.c)
    distance = abs(np.dot(incenter.a, side_a.n) - side_a.c)
    
    return gt.Circle(incenter, distance)

def incenter_t(t: gt.Triangle) -> gt.Point:
    """Returns the incenter of a triangle - the center of the incircle."""
    # Get the angle bisectors
    ab1 = angular_bisector_ppp(t.b, t.a, t.c)
    ab2 = angular_bisector_ppp(t.a, t.b, t.c)
    
    # Incenter is the intersection of angle bisectors
    return intersect_ll(ab1, ab2)

def inradius_t(t: gt.Triangle) -> gt.Measure:
    """Returns the inradius of a triangle - the radius of the incircle."""
    return radius_c(incircle_t(t))

def orthocenter_t(t: gt.Triangle) -> gt.Point:
    """Returns the orthocenter of a triangle - the intersection of the three altitudes."""
    # Create the altitudes (perpendicular lines from vertices to opposite sides)
    alt1 = orthogonal_line_pl(t.a, line_pp(t.b, t.c))
    alt2 = orthogonal_line_pl(t.b, line_pp(t.a, t.c))
    
    # Orthocenter is the intersection of the altitudes
    return intersect_ll(alt1, alt2)

def polygon_from_center_and_circumradius(num_sides: int, center: gt.Point, radius: Union[gt.Measure, float]) -> List[Union[gt.Polygon, gt.Point]]:
    """Create a regular polygon with specified number of sides, center, and circumradius."""
    # Extract radius value from Measure object if needed
    r = radius.x if isinstance(radius, gt.Measure) else float(radius)
    
    # Generate evenly spaced points around the circle
    points = []
    phase = random.random() * 2 * np.pi # random phase shift -- we do not want the construction to depend on the orientation of the polygon.
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides + phase
        # Using polar coordinates to place points evenly
        x = center.a[0] + r * np.cos(angle)
        y = center.a[1] + r * np.sin(angle)
        points.append(gt.Point(np.array([x, y])))
    
    return points + [gt.Polygon(points)]

'''
# This function was simply more trouble than it was worth.
def rotate_polygon_about_center_by_equivalent_angle(polygon: gt.Polygon, angle: gt.Angle) -> List[Union[gt.Polygon, gt.Point]]:
    points = []
    for i in range(len(polygon.points)):  
        points.append(rotate_pap(polygon.points[i], angle, polygon.center))
    return points + [gt.Polygon(points)]
'''

def rotate_polygon_about_center(polygon: gt.Polygon, angle_measure: gt.AngleSize) -> List[Union[gt.Polygon, gt.Point]]:
    points = []
    for i in range(len(polygon.points)):  
        points.append(rotate_pAp(polygon.points[i], angle_measure, polygon.center))
    return points + [gt.Polygon(points)]

# Not special. Needs surrounding logic to sample the actual diagonal,
# due to the constraints from command.apply() on the output semantics of this function.
# This function simply has a different name from segment_pp so it can be translated differently.
def diagonal_p(p1: gt.Point, p2: gt.Point) -> gt.Segment:
    return segment_pp(p1, p2)

        

def externally_tangent_c(new_radius: int, c1: gt.Circle) -> gt.Circle:
    """Create a circle that is externally tangent to the given circle c1.
    Returns the new circle and its center point."""
    # Choose a random direction for the new circle's center
    direction = gt.random_direction()
    
    # Calculate the center position for external tangency
    # For external tangency, the distance between centers equals the sum of radii
    center_distance = c1.r + new_radius
    new_center_coords = c1.c + direction * center_distance
    new_center = gt.Point(new_center_coords)
    
    # Create the new circle
    new_circle = gt.Circle(new_center_coords, new_radius)
    
    return [new_center, new_circle]

def internally_tangent_c(new_radius: int, c1: gt.Circle) -> gt.Circle:
    """Create a circle that is internally tangent to the given circle c1.
    Returns the new circle and its center point."""
    assert(new_radius < c1.r)
    direction = gt.random_direction()
    center_distance = c1.r - new_radius
    new_center_coords = c1.c + direction * center_distance
    new_center = gt.Point(new_center_coords)
    new_circle = gt.Circle(new_center_coords, new_radius)
    return [new_center, new_circle]

def externally_tangent_cc(
    new_radius: float,
    c1: gt.Circle,
    c2: gt.Circle
) -> Tuple[gt.Point, gt.Circle]:
    """
    Create a circle of radius `new_radius` that is externally tangent to both c1 and c2.
    Returns a randomly chosen solution (center point and circle).
    """
    # unit vector from c1 to c2
    center_distance = np.linalg.norm(c2.c - c1.c)
    assert np.isclose(center_distance, c1.r + c2.r), "Circles must be externally tangent to each other"
    p = (c2.c - c1.c) / center_distance

    # curvatures
    k1 = 1.0 / c1.r
    k2 = 1.0 / c2.r
    k3 = 1.0 / new_radius

    # factor in numerator
    s = 2 * np.sqrt(k1 * k2)
    weighted_sum = k1 * c1.c + k2 * c2.c

    # two possible centers
    c_plus  = (weighted_sum + s * p) / k3
    c_minus = (weighted_sum - s * p) / k3

    # wrap as Point and Circle
    sol_plus  = (gt.Point(c_plus),  gt.Circle(c_plus,  new_radius))
    sol_minus = (gt.Point(c_minus), gt.Circle(c_minus, new_radius))

    return random.choice([sol_plus, sol_minus])

def chord_c(length: Union[gt.Measure, int], circle: gt.Circle) -> Tuple[gt.Point, gt.Point, gt.Segment]:
    """Create a chord of a given length on the circle."""
    if isinstance(length, gt.Measure):
        length = length.x
    assert 0 < length <= 2 * circle.r, "Chord length must be positive and no longer than diameter"

    # Random direction vector
    direction = gt.random_direction()  # Assume this is a unit 2D vector (np.ndarray)

    # Half of the chord vector
    half_length = length / 2
    offset = direction * half_length

    # Pick a point on the circle such that chord lies across this direction
    # Compute perpendicular offset from the center along direction, maintaining chord endpoints on circle
    height = np.sqrt(circle.r**2 - half_length**2)
    midpoint = circle.c + direction * height

    p1 = midpoint - offset
    p2 = midpoint + offset

    pt1 = gt.Point(p1)
    pt2 = gt.Point(p2)
    seg = gt.Segment(pt1, pt2)
    return pt1, pt2, seg
    
# this one is special, can only be sampled first
def equilateral_triangle(side_length: Union[gt.Measure, int]) -> gt.Triangle:
    if isinstance(side_length, gt.Measure):
        side_length = side_length.x
    assert side_length > 0, "Side length must be positive"
    pa = gt.Point(np.array([0, 0]))
    pb = gt.Point(np.array([side_length, 0]))
    pc = gt.Point(np.array([side_length / 2, side_length * np.sqrt(3) / 2]))
    return gt.Triangle(pa, pb, pc), pa, pb, pc, gt.Segment(pa, pb), gt.Segment(pb, pc), gt.Segment(pc, pa)
    
    