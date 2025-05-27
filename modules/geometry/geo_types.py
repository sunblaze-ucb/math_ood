from typing import List, Union
import numpy as np

def interpolate(start, end, alpha):
    return (1-alpha)*start + alpha*end
def a_to_cpx(a):
    return complex(*a)
def cpx_to_a(cpx):
    return np.array((cpx.real, cpx.imag))
def vector_perp_rot(vec):
    return np.array((vec[1], -vec[0]))
def random_direction():
    return cpx_to_a(np.exp(np.random.random() * 2*np.pi * 1j))
def square_norm(x):
    return np.dot(x,x)
def rotate_vec(vec, alpha):
    return cpx_to_a(np.exp(alpha*1j) * a_to_cpx(vec))

class Point:
    def __init__(self, a, x = None):
        if x is not None:
            self.x = a
            self.y = x
            self.a = np.array([a, x], dtype=np.float64)
        else:
            self.a = np.array(a, dtype=np.float64)
            self.x = self.a[0]
            self.y = self.a[1]
    def __repr__(self): return "Point({}, {})".format(self.a[0], self.a[1])
    def draw(self, cr, corners):
        cr.arc(self.a[0], self.a[1], 3, 0, 2*np.pi)
        cr.fill()
    def equivalent(self, other):
        if not isinstance(other, Point): return False
        return np.isclose(self.a, other.a).all()

    def translate(self, vec):
        self.a += vec
    def scale(self, ratio):
        self.a *= ratio
    def important_points(self):
        return [self.a]

class Line:
    def __init__(self, n, c):
        self.n = np.array(n)
        assert((self.n != 0).any())
        self.c = c
        norm = np.linalg.norm(n)
        if not np.isclose(norm, 1):
            self.n /= norm
            self.c /= norm
        self.v = vector_perp_rot(self.n)

    def translate(self, vec):
        self.c += np.dot(vec, self.n)
    def scale(self, ratio):
        self.c *= ratio
    def important_points(self):
        return [self.n*self.c]

    def __repr__(self):
        return "Line(n=({}  {}) c={})".format(self.n[0], self.n[1], self.c)

    def equivalent(self, other):
        if not isinstance(other, Line): return False
        if np.isclose(self.n, other.n).all() and np.isclose(self.c, other.c):
            return True
        if np.isclose(self.n, -other.n).all() and np.isclose(self.c, -other.c):
            return True
        return False

    def get_endpoints(self, corners):

        result = [None, None]
        boundaries = list(zip(*corners))
        if np.prod(self.n) > 0:
            boundaries[1] = boundaries[1][1], boundaries[1][0]

        for coor in (0,1):
            if self.n[1-coor] == 0: continue
            for i, bound in enumerate(boundaries[coor]):
                p = np.zeros([2])
                p[coor] = bound
                p[1-coor] = (self.c - bound*self.n[coor])/self.n[1-coor]
                if (p[1-coor] - boundaries[1-coor][0]) * (p[1-coor] - boundaries[1-coor][1]) <= 0:
                    result[i] = p

        if result[0] is None or result[1] is None: return None
        else: return result

    def random_point(self, corners):

        endpoints = self.get_endpoints(corners)
        if endpoints is None: return self.n*self.c
        return interpolate(endpoints[0], endpoints[1], np.random.random())

    def draw(self, cr, corners):
        endpoints = self.get_endpoints(corners)
        if endpoints is None: return

        cr.move_to(*endpoints[0])
        cr.line_to(*endpoints[1])

        cr.set_line_width(1)
        cr.stroke()

    def contains(self, x):
        return np.isclose(np.dot(x,self.n), self.c)

class Segment(Line):
    def __init__(self, p1: Union[Point, np.ndarray], p2: Union[Point, np.ndarray]): # [x,y] in Line([a,b],c) <=> xa + yb == c
        if isinstance(p1, Point):
            p1 = p1.a
        if isinstance(p2, Point):
            p2 = p2.a
        assert(not np.isclose(p1, p2).all())
        normal_vec = vector_perp_rot(p1-p2)
        c = np.dot(p1, normal_vec)
        Line.__init__(self, normal_vec, c)
        self.end_points = np.array([p1, p2])
        self.length = np.linalg.norm(p1-p2)

    def translate(self, vec):
        self.c += np.dot(vec, self.n)
        self.end_points += vec
    def scale(self, ratio):
        self.c *= ratio
        self.end_points *= ratio
    def important_points(self):
        return [np.average(self.end_points, axis = 0)]

    def __repr__(self):
        return self.end_points.__repr__()+":"+Line.__repr__(self)

    def get_endpoints(self, corners):
        return self.end_points

    def contains(self, x):
        if not Line.contains(self, x): return False
        p1, p2 = self.end_points
        for x in (np.dot(p2-p1, x-p1), np.dot(p1-p2, x-p2)):
            if x < 0 and not np.isclose(x,0): return False
        return True

class Ray(Line):
    def __init__(self, start_point, vec):
        normal_vec = -vector_perp_rot(vec)
        c = np.dot(start_point, normal_vec)
        Line.__init__(self, normal_vec, c)
        self.start_point = start_point

    def translate(self, vec):
        self.c += np.dot(vec, self.n)
        self.start_point += vec
    def scale(self, ratio):
        self.c *= ratio
        self.start_point *= ratio
    def important_points(self):
        return [self.start_point]

    def get_endpoints(self, corners):
        line_endpoints = Line.get_endpoints(self, corners)
        if line_endpoints is None: return None
        pos_endpoints = [
            point
            for point in line_endpoints
            if np.dot(self.v, point - self.start_point) > 0
        ]
        if len(pos_endpoints) == 0: return None
        elif len(pos_endpoints) == 1:
            return [self.start_point, pos_endpoints[0]]
        else: return pos_endpoints

    def contains(self, x):
        if not Line.contains(self, x): return False
        return np.dot(self.v, x-self.start_point) >= 0

class Angle:
    def __init__(self, p, v1, v2):
        self.p = p
        self.angle = np.angle(a_to_cpx(v2) / a_to_cpx(v1))

        if self.angle < 0:
            self.angle += 2*np.pi
        if self.angle > np.pi: # get the smaller angle
            self.angle = 2*np.pi - self.angle
            v1, v2 = v2, v1

        self.start_angle = np.angle(a_to_cpx(v1))
        self.end_angle = self.start_angle + self.angle
        self.v1 = v1
        self.v2 = v2
        max_r = min(np.linalg.norm(v1), np.linalg.norm(v2))*0.45
        self.r = min(max_r, 30 / self.angle**0.5)

    def translate(self, vec):
        self.p += vec
    def scale(self, ratio):
        self.p *= ratio
        self.v1 *= ratio
        self.v2 *= ratio
    def important_points(self):
        return [self.p]

    def __repr__(self):
        return "Angle({}°)".format(self.angle/np.pi * 180)

    def equivalent(self, other):
        if isinstance(other, Angle): return np.isclose(self.angle, other.angle)
        if isinstance(other, AngleSize): return np.isclose(self.angle, other.x)
        return False

    def draw(self, cr, corners):
        cr.arc(self.p[0], self.p[1], self.r,
               self.start_angle, self.end_angle)
        cr.set_line_width(1)
        cr.stroke()

# implicitly a regular polygon now, not in original pyggb though
class Polygon:
    def __init__(self, points: List[Point]):
        self.points = points
        self.center = Point(np.average(np.array([p.a for p in self.points]), axis = 0))

    def translate(self, vec):
        self.points += vec
    def scale(self, ratio):
        self.points *= ratio
    def important_points(self):
        return []

    def draw(self, cr, corners):

        return
        cr.save()
        cr.move_to(*self.points[-1])
        for point in self.points:
            cr.line_to(*point)
        cr.set_source_rgba(0, 0, 0, 0.1)
        cr.fill()
        cr.restore()

class Triangle:
    def __init__(self, a: Point, b: Point, c: Point):
        self.a = a
        self.b = b
        self.c = c
        self.segments = [Segment(a, b), Segment(b, c), Segment(c, a)]
        self.points = [a, b, c] # for area_P
        assert(not np.isclose(self.a.a, self.b.a).all())
        assert(not np.isclose(self.b.a, self.c.a).all())
        assert(not np.isclose(self.c.a, self.a.a).all())

    def translate(self, vec):
        self.a += vec
        self.b += vec
        self.c += vec
    def scale(self, ratio):
        self.a *= ratio
        self.b *= ratio
        self.c *= ratio
    def important_points(self):
        return [self.a, self.b, self.c]
    
    def __repr__(self):
        return "Triangle(a={}, b={}, c={})".format(self.a, self.b, self.c)
    
    def equivalent(self, other):
        return np.isclose(self.a, other.a).all() and np.isclose(self.b, other.b).all() and np.isclose(self.c, other.c).all()
    # Completely untested
    def draw(self, cr, corners):
        cr.move_to(*self.a)
        cr.line_to(*self.b)
        cr.line_to(*self.c)
        cr.close_path()
        cr.set_source_rgba(0, 0, 0, 0.1)
        cr.fill()

class Circle:
    def __init__(self, center: Union[Point, np.ndarray], r: float):
        assert(r > 1e-3)
        if isinstance(center, Point):
            self.c = center.a
        else:
            self.c = center
        self.center = Point(self.c)
        self.r = r
        self.radius = r
        self.r_squared = self.r**2

    def translate(self, vec):
        self.c += vec
    def scale(self, ratio):
        self.c *= ratio
        self.r *= ratio
    def important_points(self):
        return [self.c]

    def __repr__(self):
        return "Circle(c=({}  {}) r={})".format(self.c[0], self.c[1], self.r)

    def equivalent(self, other):
        if not isinstance(other, Circle): return False
        return np.isclose(self.c, other.c).all() and np.isclose(self.r, other.r)

    def draw(self, cr, corners):
        cr.arc(self.c[0], self.c[1], self.r, 0, 2*np.pi)
        cr.set_line_width(1)
        cr.stroke()

    def contains(self, x):
        return np.isclose(square_norm(x-self.c), self.r_squared)

class Arc(Circle):
    def __init__(self, center, r, angles):
        Circle.__init__(self, center, r)
        self.angles = [a % (2*np.pi) for a in angles]

    def important_points(self):
        p1, p2 = [
            self.c + cpx_to_a(self.r*np.exp(a*1j))
            for a in self.angles
        ]
        return [(p1+p2)/2]

    def draw(self, cr, corners):
        cr.arc(self.c[0], self.c[1], self.r, *self.angles)
        cr.set_line_width(1)
        cr.stroke()

    def contains(self, x):
        if not Circle.contains(self, x): return False
        a1,a2 = self.angles
        x_angle = np.angle(a_to_cpx(x-self.c))
        if np.isclose(x_angle, a1) or np.isclose(x_angle, a2): return True
        if a1 == a2: return False
        perm_sign = int(a1 <= a2) + int(a1 <= x_angle) + int(x_angle <= a2)
        return perm_sign%2 == 1

class Vector:
    def __init__(self, end_points):
        self.end_points = np.array(end_points)
        self.v = self.end_points[1] - self.end_points[0]
        assert(not np.isclose(self.v, np.zeros(2)).all())
    def translate(self, vec):
        self.end_points += vec
    def scale(self, ratio):
        self.end_points *= ratio
        self.v *= ratio
    def important_points(self):
        return list(self.end_points)

    def equivalent(self, other):
        if not isinstance(other, Vector): return False
        return np.isclose(self.v, other.v).all()

    def draw(self, cr, corners):
        cr.move_to(*self.end_points[0])
        cr.line_to(*self.end_points[1])
        cr.set_line_width(1)
        cr.stroke()

        tip_vec = -12*self.v / np.linalg.norm(self.v)
        tip_vec2 = 0.5*vector_perp_rot(tip_vec)
        cr.move_to(*self.end_points[1])
        cr.line_to(*self.end_points[1]+tip_vec+tip_vec2)
        cr.line_to(*self.end_points[1]+tip_vec-tip_vec2)
        cr.move_to(*self.end_points[1])
        cr.fill()

class Measure:
    def __init__(self, x, dim = 0):
        self.x = x
        self.dim = dim
    def __repr__(self):
        return "Measure{}({})".format(self.dim, self.x)

    def translate(self, vec):
        pass
    def scale(self, ratio):
        if self.dim != 0: self.x *= ratio ** self.dim

    def equivalent(self, other):
        if not isinstance(other, Measure): return False
        return np.isclose(self.x, other.x)

class AngleSize:
    def __init__(self, x):
        self.x = x
    def __repr__(self):
        return "AngleSize({}°, radians={})".format(self.x/np.pi * 180, self.x)

    def translate(self, vec):
        pass
    def scale(self, ratio):
        pass

    def equivalent(self, other):
        if isinstance(other, Angle): return np.isclose(self.x, other.angle)
        if isinstance(other, AngleSize): return np.isclose(self.x, other.x)
        return False

class Boolean:
    def __init__(self, b):
        self.b = b
    def __repr__(self):
        return "Boolean({})".format(self.b)

    def translate(self, vec):
        pass
    def scale(self, ratio):
        pass

    def equivalent(self, other):
        if not isinstance(other, Boolean): return False
        return self.b == other.b

# Define all types that can be meaningfully measured
MEASURABLE_TYPES = (
    # Numeric types
    Measure, # no good reason to directly measure float/int/anglesize bcause they only arise when we construct them directly
    # Geometric types with direct numeric value
    Angle, Segment, Polygon, Triangle
    # Remove Boolean because it makes boring constructions
    # Boolean
)
