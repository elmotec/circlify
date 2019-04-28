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

import sys
import math
import collections
import itertools
import logging


__version__ = '0.10.0'


try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as pltp

    def bubbles(elements, lim=None):
        """Debugging function displays circles with matplotlib."""
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
        for elem in elements:
            x, y, r = elem.circle
            ax.add_patch(pltp.Circle((x, y), r, alpha=0.2,
                                     linewidth=2, fill=False))
            label = elem.id_ if elem.id_ is not None else elem.datum
            ax.text(x, y, label)
        if lim is None:
            lim = max([max(abs(elem.circle.x) + elem.circle.r,
                           abs(elem.circle.y) + elem.circle.r)
                       for elem in elements])
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.show()
except ImportError:
    pass


Circle = collections.namedtuple('Circle', ['x', 'y', 'r'])
FieldNames = collections.namedtuple('Field', ['id', 'datum', 'children'])

class Element:
    """Hierarchy element.

    Used as an intermediate and output data structure.

    """

    __slots__ = ['level', 'datum', 'id_', 'children', 'circle']

    def __init__(self, level, datum, id_=None, children=None, circle=None):
        """Initialize Output data structure.

        Args:
            level (int): depth level of the data where 0 is the root of the
                hierarchy.
            datum (float): value that used the size of the circle.
            id_ (str): optional ID that can be used to identify the element.
            children (list): optional list of other Output at the lower level.
            circle (Circle): coords of the circle that represent the element.

        """
        self.level = level
        self.datum = datum
        self.id_ = id_
        self.children = children
        self.circle = circle

    def __lt__(self, other):
        """Reversed level order, then normal ordering on datum."""
        return (-self.level, self.datum) < (-other.level, other.datum)

    def __eq__(self, other):
        """Compare level and datum. No order on id, children and circle."""
        return (self.level, self.datum) == (other.level, other.datum)

    def __repr__(self):
        """Representation of Output"""
        return "{}(level={}, datum={}, id_={!r}, children={}, circle={})".\
            format(self.__class__.__name__, self.level, self.datum, self.id_,
                   self.children, self.circle)


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_eps = sys.float_info.epsilon

Circle = collections.namedtuple('Circle', ['x', 'y', 'r'])

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
    if d > r1 + r2:
        log.debug('no solution, the circles are separate: %s, %s',
                  circle1, circle2)
        return None, None
    if d < abs(r1 - r2):
        log.debug('no solution, circles contained within each other: %s, %s',
                  circle1, circle2)
        return None, None
    if d == 0 and r1 == r2:
        log.debug('no solution, circles are coincident: %s, %s',
                  circle1, circle2)
        return None, None
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / d
    ym = y1 + a* dy / d
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
    ic1 = Circle(c1.x, c1.y, c1.r + (radius + margin))
    ic2 = Circle(c2.x, c2.y, c2.r + (radius + margin))
    i1, i2 = get_intersection(ic1, ic2)
    if i1 is None:
        return None, None
    i1_x, i1_y = i1
    candidate1 = Circle(i1_x, i1_y, radius)
    if i2 is None:
        return candidate1, None
    i2_x, i2_y = i2
    candidate2 = Circle(i2_x, i2_y, radius)
    return candidate1, candidate2


def get_hole_degree(candidate, placed_circles, pc1, pc2):
    """Calculate the hole degree of a candidate circle.

    Note pc1 and pc2 must not be used in the evaluation of the minimum
    distance.

    Args:
        candidate: candidate circle.
        placed_circles: sequence of circles already placed.
        pc1: first placed circle used to place the candidate.
        pc2: second placed circle used to place the candidate.

    Returns:
        hole degree defined as (1 - d_min / r_i) where d_min is a minimum
        disance between the candidate and the circles other than the one
        used to place the candidate and r_i the radius of the candidate.

    """
    lsq = 0.
    for pc in placed_circles:
        if pc1 is not None and pc1 == pc:
            continue
        if pc2 is not None and pc2 == pc:
            continue
        lsq += math.sqrt((candidate.y - pc.y) ** 2.0 +
                         (candidate.x - pc.x) ** 2.0)
    return -math.sqrt(lsq)


def pack_A1_0(data):
    """Pack circles whose radius is linked to the input data.

    This algorithm is based on the Huang et al. heuristic.

    Args:
        data: sorted (descending) list of value to circlify.

    Returns:
        list of circlify.Output.

    """
    assert data == sorted(data, reverse=True), 'data must be sorted (desc)'
    placed_circles = []
    for value in data:
        radius = math.sqrt(value)
        n_circles = len(placed_circles)
        # Place first 2 circles on each side of (0, 0)
        if n_circles <= 1:
            x = radius if n_circles == 0 else -radius
            circle = Circle(x, float(0.0), radius)
            placed_circles.append(circle)
            continue
        mhd = None
        lead_candidate = None
        for (c1, c2) in itertools.combinations(placed_circles, 2):
            margin = float(_eps)
            for cand in get_placement_candidates(radius, c1, c2, margin):
                if cand is not None:
                    # Ignore candidates that overlap with any placed circle.
                    other_placed_circles = [circ for circ in placed_circles
                                            if circ not in [c1, c2]]
                    overlap = False
                    for pc in other_placed_circles:
                        dist =  distance(pc, cand)
                        if dist < -_eps:
                            overlap = True
                            break
                    if overlap:
                        continue
                    hd = get_hole_degree(cand, placed_circles, c1, c2)
                    if mhd is None or hd > mhd:
                        mhd = hd
                        lead_candidate = cand
        if lead_candidate is None:
            raise ValueError('cannot place circle for value %f', value)
        placed_circles.append(lead_candidate)
    return placed_circles


def extendBasis(B, p):
    """Extend basis to ...  """
    if enclosesWeakAll(p, B):
        return [p];

    # If we get here then B must have at least one element.
    for i in range(len(B)):
        if enclosesNot(p, B[i]) and enclosesWeakAll(encloseBasis2(B[i], p), B):
            return [B[i], p];

    # If we get here then B must have at least two elements.
    for i in range(len(B) - 1):
        for j in range(i + 1, len(B)):
            if enclosesNot(encloseBasis2(B[i], B[j]), p) and \
                    enclosesNot(encloseBasis2(B[i], p), B[j]) and \
                    enclosesNot(encloseBasis2(B[j], p), B[i]) and \
                    enclosesWeakAll(encloseBasis3(B[i], B[j], p), B):
                return [B[i], B[j], p];
    raise RuntimeError('If we get here then something is very wrong')


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
    for i in range(len(B)):
        if not enclosesWeak(a, B[i]):
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
    l = math.sqrt(x21 * x21 + y21 * y21);
    return Circle((x1 + x2 + x21 / l * r21) / 2,
                  (y1 + y2 + y21 / l * r21) / 2,
                  (l + r1 + r2) / 2)


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
    return Circle(x1 + xa + xb * r, y1 + ya + yb * r, r)


def scale(circles, target, enclosure=None):
    """Scale circles in enclosure to fit in the target circle.

    Args:
        circles: Circle objects to scale.
        target: target Circle to scale to.
        enclosure: allows one to specify the enclosure.

    Returns:
        scaled circle

    """
    if not circles:
        return circles
    if not enclosure:
        enclosure = enclose(circles)
    if enclosure is None:
        raise ValueError('cannot enclose circles')
    r = target.r / enclosure.r
    t_x, t_y = target.x, target.y
    e_x, e_y = enclosure.x, enclosure.y
    c_x, c_y, c_r = circle
    return Circle((c_x - e_x) * r + t_x, (c_y - e_y) * r + t_y, c_r * r)


def enclose(circles):
    """Shamelessly adapted from d3js.

    See https://github.com/d3/d3-hierarchy/blob/master/src/pack/enclose.js

    """
    B = []
    p, e = None, None
    #random.shuffle(circles)

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


def _handle(data, level, fields=None):
    """Converts possibly heterogeneous list of float or dict in list of Output.

    Return:
        list of list of Output. There is one list per level and the (level
        specific) sub-list sorts data by descending order.

    """
    if fields is None:
        fields = FieldNames(None, None, None)
    datum_field = fields.datum if fields.datum else 'datum'
    id_field = fields.id if fields.id else 'id'
    child_field = fields.children if fields.children else 'children'
    elements = []
    for datum in data:
        try:  # try to leverage element as a numeric value.
            elements.append(Element(level, datum + 0, None, None, None))
        except TypeError:  # if it fails, assume dict.
            if not isinstance(datum, dict):
                raise TypeError('dict or numeric value expected')
            value = datum[datum_field]
            id_ = datum[id_field] if id_field in datum else None
            children = _handle(datum[child_field], level + 1, fields) \
                if child_field in datum else None
            elements.append(Element(level, value, id_, children, None))
    return sorted(elements, reverse=True)


def _circlify_level(data, target_enclosure, fields, with_enclosure=True):
    """Pack and enclose circles whose radius is linked to the input data.

    All the elements of data are expected to be for the same parent circle
    called enclosure.

    Args:
        data (list of Element): structured data to be process.
        target_enclosure (Circle): target enclosure to fit the cirlces into.
        fields (FieldNames): field names.
        with_enclosure (bool):

    Returns:
        list of circlify.Output as value for element of data.

    """
    packed = pack_A1_0(data)
    if target_enclosure is None:
        target_enclosure = Circle(float(0.0), float(0.0), float(1.0))
    packed_and_scaled = scale(packed, target_enclosure)
    if with_enclosure:
        packed_and_scaled.append(target_enclosure)
    for circle, elem in zip(packed, data):
        elem.circle = scale(circle, target_enclosure)
        elem.children = _circlify_level(elem.children, elem.circle, fields)
    return data


def _flatten(elements, flattened):
    """Flattens the elements hierarchy."""
    if elements is None:
        return
    for elem in elements:
        _flatten(elem.children, flattened)
        elem.children = None
        flattened.append(elem)
    return flattened


def circlify(data, target_enclosure=None, show_enclosure=False,
             datum_field='datum', id_field='id', children_field='children'):
    """Pack and enclose circles whose radius is linked to the input data.

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
        list of circlify.Element sorted by ascending level (root to leaf) and
        descending value (biggest circles first).

    """
    fields = FieldNames(id=id_field, datum=datum_field, children=children_field)
    elements = _handle(data, 1, fields)
    if target_enclosure is None:
        target_enclosure = Circle(0.0, 0.0, 1.0)
    elements = _circlify_level(elements, target_enclosure, fields)
    if show_enclosure:
        elements.append(Element(0, None, circle=target_enclosure))
    flattened = []
    return sorted(_flatten(elements, flattened), reverse=True)
