#!/usr/bin/env python
# encoding: utf-8

"""Basic circle packing algorithm based on 2 algorithms.

Circles are first arranged via a version of A1.0 by Huang et al (see
https://home.mis.u-picardie.fr/~cli/Publis/circle.pdf for details) and then
enclosed in a circle created around them using Matoušek-Sharir-Welzl algorithm
used in d3js (see https://beta.observablehq.com/@mbostock/miniball,
http://www.inf.ethz.ch/personal/emo/PublFiles/SubexLinProg_ALG16_96.pdf, and
https://github.com/d3/d3-hierarchy/blob/master/src/pack/enclose.js)

"""

import collections
import itertools
import logging
import math
import sys

__version__ = "0.14.0"


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_eps = sys.float_info.epsilon


try:  # pragma: no cover  # noqa
    import matplotlib.pyplot as plt

    def get_default_label(count, circle):
        """Generates a default label."""
        if circle.ex and "id" in circle.ex:
            label = str(circle.ex["id"])
        elif circle.ex and "datum" in circle.ex:
            label = circle.ex["datum"]
        elif circle.ex:
            label = str(circle.ex)
        else:
            label = "#" + str(count)
        return label

    def _bubbles(circles, labels=None, lim=None):
        """Debugging function displays circles with matplotlib."""
        if not labels:
            labels = range(len(circles))
        _, ax = plt.subplots(figsize=(8.0, 8.0))
        for circle, label in zip(circles, labels):
            x, y, r = circle
            ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2, fill=False))
            ax.text(x, y, label)
        enclosure = enclose(circles)
        n = len(circles)
        if enclosure in circles:
            n = n - 1
        d = density(circles, enclosure)
        title = "{} circles packed for density {:.4f}".format(n, d)
        ax.set_title(title)
        if lim is None:
            lim = max(
                max(
                    abs(circle.x) + circle.r,
                    abs(circle.y) + circle.r,
                )
                for circle in circles
            )
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.show()

    def bubbles(circles, labels=None, lim=None):
        if not labels:
            labels = [get_default_label(i, c) for i, c in enumerate(circles)]
        return _bubbles([c.circle for c in circles], labels, lim)

except ImportError:  # pragma: no cover
    pass


_Circle = collections.namedtuple("_Circle", ["x", "y", "r"])
FieldNames = collections.namedtuple("Field", ["id", "datum", "children"])


class Circle:
    """Hierarchy element.

    Used as an intermediate and output data structure.

    """

    __slots__ = ["circle", "level", "ex"]

    def __init__(self, x=0.0, y=0.0, r=1.0, level=1, ex=None):
        """Initialize Output data structure.

        Args:
            x (float): x coordinate of the circle center.
            y (float): y coordinate of the circle center.
            r (float): radius of the circle.
            level (int): depth level of the data for hierarchy representation
                where 0 is the root of the hierarchy.
            ex (dict): additional data to be carried by the structure (e.g.
                label, name, parent_id, ...)

        """
        self.circle = _Circle(x, y, r)
        self.level = level
        self.ex = ex

    def __lt__(self, other):
        """Reversed level order, then normal ordering on datum."""
        return (self.level, self.r) < (other.level, other.r)

    def __eq__(self, other):
        """Compare level and datum. No order on id, children and circle."""
        return (self.level, self.circle, self.ex) == (
            other.level,
            other.circle,
            other.ex,
        )

    def __repr__(self):
        """Representation of Output"""
        return "{}(x={}, y={}, r={}, level={}, ex={!r})".format(
            self.__class__.__name__, self.x, self.y, self.r, self.level, self.ex
        )

    def __iter__(self):
        """Convenience function to unpack circle in triple (x, y, r)"""
        return [self.x, self.y, self.r].__iter__()

    @property
    def x(self):
        return self.circle.x

    @property
    def y(self):
        return self.circle.y

    @property
    def r(self):
        return self.circle.r


def distance(circle1, circle2):
    """Compute distance between 2 cirlces."""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    x = x2 - x1
    y = y2 - y1
    return math.sqrt(x * x + y * y) - r1 - r2


def get_intersection(circle1, circle2):
    """Calculate intersections of 2 circles

    Based on https://gist.github.com/xaedes/974535e71009fa8f090e
    Credit to http://stackoverflow.com/a/3349134/798588

    Returns:
        2 pairs of coordinates. Each pair can be None if there are no or just
        one intersection.

    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    dx, dy = x2 - x1, y2 - y1
    d = math.sqrt(dx * dx + dy * dy)
    # Protect this part of the algo with try/except because edge cases
    # can lead to divizion by 0 or sqrt of negative numbers. Those indicate
    # that no intersection can be found and the debug log will show more info.
    try:
        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        h = math.sqrt(r1 * r1 - a * a)
    except (ValueError, ZeroDivisionError):
        eps = 1e-9
        if d > r1 + r2:
            log.debug("no solution, the circles are separate: %s, %s", circle1, circle2)
        if d < abs(r1 - r2) + eps:
            log.debug(
                "no solution, circles contained within each other: %s, %s",
                circle1,
                circle2,
            )
        if math.isclose(d, 0, abs_tol=eps) and math.isclose(
            r1, r2, rel_tol=0.0, abs_tol=eps
        ):
            log.debug("no solution, circles are coincident: %s, %s", circle1, circle2)
        return None, None
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d
    if xs1 == xs2 and ys1 == ys2:
        return (xs1, ys1), None
    return (xs1, ys1), (xs2, ys2)


def get_placement_candidates(radius, c1, c2, margin):
    """Generate placement candidates for 2 existing placed circles.

    Args:
        radius: radius of the new circle to place.
        c1: first existing placed circle.
        c2: second existing placed circle.
        margin: extra padding between the candidates and existing c1 and c2.

    Returns:
        A pair of candidate cirlces where one or both value can be None.

    """
    margin = radius * _eps * 10.0
    ic1 = _Circle(c1.x, c1.y, c1.r + (radius + margin))
    ic2 = _Circle(c2.x, c2.y, c2.r + (radius + margin))
    i1, i2 = get_intersection(ic1, ic2)
    if i1 is None:
        return None, None
    i1_x, i1_y = i1
    candidate1 = _Circle(i1_x, i1_y, radius)
    if i2 is None:
        return candidate1, None
    i2_x, i2_y = i2
    candidate2 = _Circle(i2_x, i2_y, radius)
    return candidate1, candidate2


def get_hole_degree_radius_w(candidate, circles):
    """Calculate the hole degree of a candidate circle.

    Args:
        candidate: candidate circle.
        circles: sequence of circles.

    Returns:
        Squared euclidian distance of the candidate to the circles in argument.
        Each component of the distance is weighted by the inverse of the radius
        of the other circle to tilt the choice towards bigger circles.

    """
    return sum(distance(candidate, c) * c.r for c in circles)


def get_hole_degree_a1_0(candidate, circles):
    """Calculate the hole degree of a candidate circle.

    Args:
        candidate: candidate circle.
        circles: sequence of circles.

    Returns:
        minimum distance between the candidate and the circles in argument.

    In the paper, the hole degree defined as (1 - d_min / r_i) where d_min is
    a minimum disance between the candidate and the circles other than the one
    used to place the candidate and r_i the radius of the candidate.

    """
    return min(distance(candidate, c) for c in circles)


def get_hole_degree_density(candidate, circles):
    """Calculate the hole degree of a candidate circle.

    Args:
        candidate: candidate circle.
        circles: sequence of circles.

    Returns:
        One minus the density of the configuration. So the result should always
        be between 0 and 1. See also density.

    """
    return 1.0 - density(circles + [candidate])


def place_new_A1_0(radius, next_, const_placed_circles, get_hole_degree):
    """Place a new circle.

    Args:
        radius: value to be added.
        next_: next value to be added after radius.
        const_placed_circles: circles.
        get_hole_degree: objective function to maximize.

    Returns:
        New configuration.

    """
    placed_circles = const_placed_circles[:]
    n_circles = len(placed_circles)
    # If there are 2 or less, place circles on each side of (0, 0)
    if n_circles <= 1:
        x = radius if n_circles == 0 else -radius
        circle = _Circle(x, float(0.0), radius)
        placed_circles.append(circle)
        return placed_circles
    mhd = None
    lead_candidate = None
    for (c1, c2) in itertools.combinations(placed_circles, 2):
        margin = radius * _eps * 10.0
        # Placed circles other than the 2 circles used to find the
        # candidate placement.
        other_circles = [c for c in placed_circles if c not in (c1, c2)]
        for cand in get_placement_candidates(radius, c1, c2, margin):
            if cand is None:
                continue
            if not other_circles:
                lead_candidate = cand
                break
            # If overlaps with any, skip this candidate.
            if any(distance(c, cand) < 0.0 for c in other_circles):
                continue
            hd = get_hole_degree(cand, other_circles)
            assert hd is not None, "hole degree should not be None!"
            # If we were to use next_ we could use it here for look ahead.
            if mhd is None or hd < mhd:
                mhd = hd
                lead_candidate = cand
            if abs(mhd) < margin:
                break
    if lead_candidate is None:
        # The radius is set to sqrt(value) in pack_A1_0
        raise ValueError("cannot place circle for value " + str(radius**2))
    placed_circles.append(lead_candidate)
    return placed_circles


def pack_A1_0(data):
    """Pack circles whose area is proportional to the input data.

    This algorithm is based on the Huang et al. heuristic.

    Args:
        data: sorted (descending) list of value to circlify.

    Returns:
        list of circlify.Output.

    """
    min_max_ratio = min(data) / max(data)
    if min_max_ratio < _eps:
        log.warning(
            "min to max ratio is too low at %f and it could cause algorithm stability issues. Try to remove insignificant data",
            min_max_ratio,
        )
    assert data == sorted(data, reverse=True), "data must be sorted (desc)"
    placed_circles = []
    radiuses = [math.sqrt(value) for value in data]
    for radius, next_ in look_ahead(radiuses):
        placed_circles = place_new_A1_0(
            radius, next_, placed_circles, get_hole_degree_radius_w
        )
    return placed_circles


def extendBasis(B, p):
    """Extend basis to ..."""
    if enclosesWeakAll(p, B):
        return [p]

    # If we get here then B must have at least one element.
    for bel in B:
        if enclosesNot(p, bel) and enclosesWeakAll(encloseBasis2(bel, p), B):
            return [bel, p]

    # If we get here then B must have at least two elements.
    for i in range(len(B) - 1):
        for j in range(i + 1, len(B)):
            if (
                enclosesNot(encloseBasis2(B[i], B[j]), p)
                and enclosesNot(encloseBasis2(B[i], p), B[j])
                and enclosesNot(encloseBasis2(B[j], p), B[i])
                and enclosesWeakAll(encloseBasis3(B[i], B[j], p), B)
            ):
                return [B[i], B[j], p]
    raise ValueError("If we get here then something is very wrong")


def enclosesNot(a, b):
    dr = a.r - b.r
    dx = b.x - a.x
    dy = b.y - a.y
    return dr < 0 or dr * dr < dx * dx + dy * dy


def enclosesWeak(a, b):
    dr = a.r - b.r + 1e-6
    dx = b.x - a.x
    dy = b.y - a.y
    return dr > 0 and dr * dr > dx * dx + dy * dy


def enclosesWeakAll(a, B):
    for bel in B:
        if not enclosesWeak(a, bel):
            return False
    return True


def encloseBasis(B):
    if len(B) == 1:
        return B[0]
    elif len(B) == 2:
        return encloseBasis2(B[0], B[1])
    else:
        return encloseBasis3(B[0], B[1], B[2])


def encloseBasis2(a, b):
    x1, y1, r1 = a.x, a.y, a.r
    x2, y2, r2 = b.x, b.y, b.r
    x21 = x2 - x1
    y21 = y2 - y1
    r21 = r2 - r1
    l21 = math.sqrt(x21 * x21 + y21 * y21)
    return _Circle(
        (x1 + x2 + x21 / l21 * r21) / 2,
        (y1 + y2 + y21 / l21 * r21) / 2,
        (l21 + r1 + r2) / 2,
    )


def encloseBasis3(a, b, c):
    x1, y1, r1 = a.x, a.y, a.r
    x2, y2, r2 = b.x, b.y, b.r
    x3, y3, r3 = c.x, c.y, c.r
    a2 = x1 - x2
    a3 = x1 - x3
    b2 = y1 - y2
    b3 = y1 - y3
    c2 = r2 - r1
    c3 = r3 - r1
    d1 = x1 * x1 + y1 * y1 - r1 * r1
    d2 = d1 - x2 * x2 - y2 * y2 + r2 * r2
    d3 = d1 - x3 * x3 - y3 * y3 + r3 * r3
    ab = a3 * b2 - a2 * b3
    xa = (b2 * d3 - b3 * d2) / (ab * 2) - x1
    xb = (b3 * c2 - b2 * c3) / ab
    ya = (a3 * d2 - a2 * d3) / (ab * 2) - y1
    yb = (a2 * c3 - a3 * c2) / ab
    A = xb * xb + yb * yb - 1
    B = 2 * (r1 + xa * xb + ya * yb)
    C = xa * xa + ya * ya - r1 * r1
    if A != 0.0:
        r = -(B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    else:
        r = -C / B
    return _Circle(x1 + xa + xb * r, y1 + ya + yb * r, r)


def enclose(circles):
    """Shamelessly adapted from d3js.

    See https://github.com/d3/d3-hierarchy/blob/master/src/pack/enclose.js

    """
    B = []
    p, e = None, None
    # random.shuffle(circles)

    n = len(circles)
    i = 0
    while i < n:
        p = circles[i]
        if e is not None and enclosesWeak(e, p):
            i = i + 1
        else:
            B = extendBasis(B, p)
            e = encloseBasis(B)
            i = 0
    return e


def scale(circle, target, enclosure):
    """Scale circle in enclosure to fit in the target circle.

    Args:
        circle: Circle to scale.
        target: target Circle to scale to.
        enclosure: allows one to specify the enclosure.

    Returns:
        scaled circle

    """
    r = target.r / enclosure.r
    t_x, t_y = target.x, target.y
    e_x, e_y = enclosure.x, enclosure.y
    c_x, c_y, c_r = circle
    return _Circle((c_x - e_x) * r + t_x, (c_y - e_y) * r + t_y, c_r * r)


def density(circles, enclosure=None):
    """Computes the relative density of te packed circles.

    Args:
        circles: packed list of circles.

    Return:
        Compute the enclosure if not passed as argument and calculates the
        relative area of the sum of the inner cirlces to the area of the
        enclosure.

    """
    if not enclosure:
        enclosure = enclose(circles)
    return sum(c.r**2.0 for c in circles if c != enclosure) / enclosure.r**2.0


def look_ahead(iterable, n_elems=1):
    """Fetch look ahead elements of data

    From https://stackoverflow.com/questions/4197805/python-for-loop-look-ahead

    """
    items, nexts = itertools.tee(iterable, 2)
    nexts = itertools.islice(nexts, n_elems, None)
    return itertools.zip_longest(items, nexts)


def _handle(data, level, fields=None):
    """Converts possibly heterogeneous list of float or dict in list of Output.

    Return:
        list of list of Output. There is one list per level and the (level
        specific) sub-list sorts data by descending order.

    """
    if fields is None:
        fields = FieldNames(None, None, None)
    datum_field = fields.datum if fields.datum else "datum"
    elements = []
    for datum in data:
        if isinstance(datum, dict):
            value = datum[datum_field]
            elements.append(Circle(r=value + 0, level=level, ex=datum))
            continue
        if datum <= 0.0:
            raise ValueError("input data must be positive. Found " + str(datum))
        if datum <= _eps:
            log.warning(
                "input data %f is small and could cause stability issues. Can you scale the data set up or drop insignificant elements?",
                datum,
            )
        try:
            elements.append(Circle(r=datum + 0, level=level, ex={"datum": datum}))
        except TypeError:  # if it fails, assume dict.
            raise TypeError("dict or numeric value expected")
    return sorted(elements, reverse=True)


def _circlify_level(data, target_enclosure, fields, level=1):
    """Pack and enclose circles whose radius is linked to the input data.

    All the elements of data are expected to be for the same parent circle
    called enclosure.

    Args:
        elements (list of Circle): structured data to be process.
        target_enclosure (Circle): target enclosure to fit the cirlces into.
        fields (FieldNames): field names.
        level (int): level of depth in the hierarchy.

    Returns:
        list of circlify.Output as value for element of data.

    """
    all_circles = []
    if not data:
        return all_circles
    circles = _handle(data, 1, fields)
    packed = pack_A1_0([circle.r for circle in circles])
    enclosure = enclose(packed)
    assert enclosure is not None
    for circle, inner_circle in zip(circles, packed):
        circle.level = level
        circle.circle = scale(inner_circle, target_enclosure, enclosure)
        if circle.ex and fields.children in circle.ex:
            all_circles += _circlify_level(
                circle.ex[fields.children], circle.circle, fields, level + 1
            )
        all_circles.append(circle)
    return all_circles


def _flatten(elements, flattened):
    """Flattens the elements hierarchy."""
    if elements is None:
        return
    for elem in elements:
        _flatten(elem.children, flattened)
        elem.children = None
        flattened.append(elem)
    return flattened


def circlify(
    data,
    target_enclosure=None,
    show_enclosure=False,
    datum_field="datum",
    id_field="id",
    children_field="children",
):
    """Pack and enclose circles.

    Args:
        data: sorted (descending) array of values.
        target_enclosure: target circlify.Circle where circles should fit in.
            Defaults to unit circle centered on (0, 0).
        show_enclosure: insert the target enclosure to the output if True.
        datum_field: field name that contains the float value when the element is
            a dict.
        id_field: field name that contains the id when the element is a dict.
        children_field: field name that contains the children list when the
            element is a dict.

    Returns:
        list of circlify.Circle whose *area* is proportional to the
        corresponding input value.  The list is sorted by ascending level
        (root to leaf) and descending value (biggest circles first).

    """
    fields = FieldNames(id=id_field, datum=datum_field, children=children_field)
    if target_enclosure is None:
        target_enclosure = Circle(level=0, x=0.0, y=0.0, r=1.0)
    all_circles = _circlify_level(data, target_enclosure, fields)
    if show_enclosure:
        all_circles.append(target_enclosure)
    return sorted(all_circles)
