#!/usr/bin/env python
# encoding: utf-8

"""Tests for circlify"""

import unittest

import circlify as circ

# Set this variable to True to get a display of the layout (req matlplotlib)
display_layout = False


def density(solution):
    """Report density of the solution

    The surface of the circle relative to the enclosure

    """
    circ.scale(solution, circ.Circle(0.0, 0.0, 1.0))
    density = sum([c.r * c.r for c in solution if c.r < 1.0])
    return density

class ElementTest(unittest.TestCase):
    """Check behaviour of Element class"""

    def test_repr_and_str_with_id_(self):
        """Check str() and repr() conversion"""
        expected = "Element(level=1, datum=2.0, id_='3', " \
                   "children=None, circle=None)"
        self.assertEqual(expected, str(Element(1, 2.0, '3')))

    def test_repr_and_str_with_circle(self):
        """Check str() and repr() conversion"""
        expected = "Element(level=1, datum=2.0, id_=None, " \
                   "children=None, circle=Circle(x=0, y=0, r=1))"
        self.assertEqual(expected, str(Element(1, 2.0, circle=Circle(0, 0, 1))))


class SpecialCases(unittest.TestCase):
    """Hedge cases and obvious answers handling."""

    def test_empty_input(self):
        """What do we do when the output is empty?"""
        actual = circ.circlify([])
        self.assertEqual(actual, [])

    def test_single_value(self):
        """If there is only one value, it should occupy the whole circle."""
        actual = circ.circlify([2.0])
        expected = [Element(1, 2.0, None, None, Circle(0.0, 0.0, 1.0))]
        self.assertEqual(actual, expected)

    def test_two_equal_values(self):
        """Two equal circle cases is also trivial."""
        # Force scaling to .5 so that each circle radius is brought to 0.5.
        actual = circlify([1.0, 1.0])
        expected = [Element(1, 1.0, None, None, Circle(0.5, 0.0, 0.5)),
                    Element(1, 1.0, None, None, Circle(-0.5, 0.0, 0.5))]
        self.assertEqual(actual, expected)


class TestCaseWithDisplay(unittest.TestCase):
    """Display the result of the placement of the circle."""

    def display(self, circles, labels):
        """Forwards call to circ.circlify.bubbles()."""
        try:
            if display_layout:
                circ.bubbles(circles, labels)
        except AttributeError as err:
            print("%s. %s".format(err, "Did you install matplotlib?"))
            raise


class PrimeSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of prime number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, 3, ... up to 19."""
        self.data = [19, 17, 13, 11, 7, 5, 3, 2, 1]

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circ.circlify(self.data, with_enclosure=True)
        expected = [
            circ.Circle(x=0.3577687934670487, y=-0.13064957525245913,
                        r=0.3952921604820124),
            circ.Circle(x=-0.4114323178203371, y=-0.13064957525245913,
                        r=0.37390895080537345),
            circ.Circle(x=-0.04661299415374855, y=0.46780144257676565,
                        r=0.3269738922300244),
            circ.Circle(x=-0.04588460789059142, y=-0.6977206243364218,
                        r=0.30077223534410524),
            circ.Circle(x=-0.6132109517981927, y=0.4490810687795326,
                        r=0.2399332412600769),
            circ.Circle(x=0.4829661488722883, y=0.45417231957823834,
                        r=0.20278059970175766),
            circ.Circle(x=0.3252787490004198, y=0.7776370388468008,
                        r=0.15707317711577198),
            circ.Circle(x=-0.40283175658099657, y=0.7512387781681531,
                        r=0.128249712070483),
            circ.Circle(x=0.09222041925800772, y=0.8617116738294699,
                        r=0.09068624109026073),
            circ.Circle(x=0.0, y=0.0, r=1.0)
        ]
        self.display(actual, [str(v) for v in self.data])
        self.assertEqual(expected, actual)


class CountSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, ..."""
        self.data = list(range(7, 1, -1))

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circ.circlify(self.data, with_enclosure=True)
        expected = [
            circ.Circle(x=0.5824456027453087, y=-0.08515409741642652,
                        r=0.411362505047332),
            circ.Circle(x=-0.2097645777676309, y=-0.08515409741642652,
                        r=0.38084767546560755),
            circ.Circle(x=0.15769153632817068, y=0.54389787930532,
                        r=0.3476647713765335),
            circ.Circle(x=0.1591053210788781, y=-0.6704181394216174,
                        r=0.3109608248719408),
            circ.Circle(x=-0.45861847805947203, y=0.5154819840108332,
                        r=0.26929997392086463),
            circ.Circle(x=-0.7680630545906644, y=0.13661056172475608,
                        r=0.2198825079503118),
            circ.Circle(x=0.0, y=0.0, r=1.0)]
        self.display(actual, [str(v) for v in self.data])
        self.assertEqual(expected, actual)


class GeometricSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, ..."""
        self.data = sorted([2 ** n for n in range(4, 12)], reverse=True)

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circ.circlify(self.data, with_enclosure=True)
        self.display(actual, [str(v) for v in self.data])
        expected = [
            circ.Circle(x=0.4142135623730951, y=0.0, r=0.5857864376269051),
            circ.Circle(x=-0.5857864376269051, y=0.0, r=0.4142135623730951),
            circ.Circle(x=-0.2218254069479773, y=0.6062444788590926,
                        r=0.29289321881345254),
            circ.Circle(x=-0.20710678118654763, y=-0.492585715504708,
                        r=0.20710678118654754),
            circ.Circle(x=0.10281914590763117, y=-0.6627207198830358,
                        r=0.14644660940672627),
            circ.Circle(x=-0.5170830169797045, y=-0.5131885205024094,
                        r=0.10355339059327377),
            circ.Circle(x=-0.4276285585587573, y=-0.6656611405645536,
                        r=0.07322330470336313),
            circ.Circle(x=-0.18045635173699437, y=-0.22990093891844118,
                        r=0.051776695296636886),
            circ.Circle(x=0.0, y=0.0, r=1.0)
        ]
        self.assertEqual(expected, actual)


class EnclosureScalingTestCase(unittest.TestCase):
    """Test circ.scale function"""

    def test_simple_zoom(self):
        """Trivial zoom test when the enclosure is the same as the circle."""
        input = circ.Circle(0, 0, 0.5)
        target = circ.Circle(0, 0, 1.0)
        actual = circ.scale([input], target)
        self.assertEqual([target], actual)

    def test_simple_zoom_off_center(self):
        """Zoom test with off center circle equal to enclosure."""
        input = circ.Circle(0.5, 0.5, 0.5)
        target = circ.Circle(0.5, 0.5, 1.0)
        actual = circ.scale([input], target)
        self.assertEqual([target], actual)

    def test_simple_zoom_and_translation(self):
        """Pan and zoom test with off center circle equal to enclosure."""
        input = circ.Circle(0.5, 0.5, 0.5)
        target = circ.Circle(-0.5, 0, 1.0)
        actual = circ.scale([input], target)
        self.assertEqual([target], actual)

    def test_zoom_with_enclosure(self):
        """Zoom test with off center circle and specific enclosure"""
        input = circ.Circle(1.0, 0.0, 1.0)
        target = circ.Circle(0.0, 0.0, 1.0)
        enclosure = circ.Circle(0.0, 0.0, 2.0)
        actual = circ.scale([input], target, enclosure=enclosure)
        expected = circ.Circle(0.5, 0.0, 0.5)
        self.assertEqual([expected], actual)


class HandleDataTestCase(unittest.TestCase):
    """Test circlify._handle function."""

    def test_integer(self):
        """handles integer"""
        actual = _handle([42], 1)
        self.assertEqual([Element(1, 42, None, None, None)], actual)

    def test_float(self):
        """Handles float."""
        actual = _handle([42.0], 1)
        self.assertEqual([Element(1, 42, None, None, None)], actual)

    def test_dict_w_datum_only(self):
        """Handles dict with just the data"""
        actual = _handle([{'datum': 42}], 1)
        self.assertEqual([Element(1, 42, None, None, None)], actual)

    def test_dict_w_datum_and_id(self):
        """Handles dict with data and an id"""
        actual = _handle([{'datum': 42, 'id': '42'}], 1)
        self.assertEqual([Element(1, 42, '42', None, None)], actual)

    def test_bad_value_raise_error(self):
        """A set of non-dict, non-numeric input raises ValueError."""
        with self.assertRaises(TypeError):
            _handle({'datum', 42}, 1)

    def test_bad_dict_keys_raise_error(self):
        """A dict with the wrong key raises ValueError."""
        with self.assertRaises(TypeError):
            _handle({'datatum': 42}, 1)

    def test_handle_children(self):
        """A dict that has children."""
        actual = _handle([{'datum': 42, 'children': [1, 2]}], 1)
        expected = [
            Element(1, 42, None, children=[
                Element(2, 2, None, None, None),
                Element(2, 1, None, None, None)], circle=None),
        ]
        self.assertEqual(expected, actual)


class MultiLevelInputTestCase(TestCaseWithDisplay):
    """Handles multi-layer input."""

    def setUp(self):
        """Sets up the test case."""
        self.data = [
            0.05,
            {'id': 'a2', 'datum': 0.05},
            {'id': 'a0', 'datum': 0.8,
             'children': [0.3, 0.2, 0.2, 0.1], },
            {'id': 'a1', 'datum': 0.1,
             'children': [{'id': 'a1_1', 'datum': 0.05},
                          {'datum': 0.04},
                          0.01], },
        ]

    def test_json_input(self):
        """Simple json data."""
        actual = circlify(self.data, show_enclosure=True)
        expected = [
            Element(level=0, datum=None, circle=Circle(x=0.0, y=0.0, r=1.0)),
            Element(level=1, datum=0.8, id_='a0',
                    circle=Circle(x=0.2612038749637414, y=0.0,
                                  r=0.7387961250362586)),
            Element(level=1, datum=0.1, id_='a1',
                    circle=Circle(x=-0.7387961250362587, y=0.0,
                                  r=0.2612038749637415)),
            Element(level=1, datum=0.05,
                    circle=Circle(x=-0.565803075997749, y=0.41097786651145324,
                                  r=0.18469903125906464)),
            Element(level=1, datum=0.05, id_='a2',
                    circle=Circle(x=-0.3385727489559141, y=0.7022188441650276,
                                  r=0.18469903125906464)),
            Element(level=2, datum=0.3,
                    circle=Circle(x=0.5533243963620484, y=-0.230392881357073,
                                  r=0.36675408601105247)),
            Element(level=2, datum=0.2,
                    circle=Circle(x=-0.11288314691830154, y=-0.230392881357073,
                                  r=0.2994534572692975)),
            Element(level=2, datum=0.2,
                    circle=Circle(x=0.15631936804871832, y=0.30460197676548245,
                                  r=0.2994534572692975)),
            Element(level=2, datum=0.1,
                    circle=Circle(x=0.6664952237042423, y=0.3369290873460549,
                                  r=0.2117455702848763)),
            Element(level=2, datum=0.05, id_='a1_1',
                    circle=Circle(x=-0.6154723840806618, y=0.0,
                                  r=0.13788013400814464)),
            Element(level=2, datum=0.04,
                    circle=Circle(x=-0.8766762590444033, y=0.0,
                                  r=0.1233237409555968)),
            Element(level=2, datum=0.01,
                    circle=Circle(x=-0.7567888163564136, y=0.14087823651338607,
                                  r=0.0616618704777984))
        ]
        self.display(actual)
        self.assertEqual(expected, actual)

    def test_handle_single_value(self):
        """Typical specification of data with just a value."""
        actual = circlify([self.data[0]])
        expected = [Element(1, 0.05, None, None, Circle(0, 0, 1))]
        self.assertEqual(expected, actual)

    def test_handle_custom_datum_key(self):
        """Specify value as dict with custom keys."""
        actual = circlify([{'value': 0.05}], datum_field='value')
        expected = [Element(1, 0.05, None, None, Circle(0, 0, 1))]
        self.assertEqual(expected, actual)

    def test_handle_custom_id_key(self):
        """Specify value as dict with custom keys."""
        actual = circlify([{'name': 'a2', 'datum': 0.05}], id_field='name')
        expected = [Element(1, 0.05, 'a2', None, Circle(0, 0, 1))]
        self.assertEqual(expected, actual)

    def test_handle_dict(self):
        """Specify value as a dict."""
        actual = circlify([self.data[1]])
        expected = [Element(1, 0.05, 'a2', None, Circle(0, 0, 1))]
        self.assertEqual(expected, actual)

    def test_handle_dict_w_children(self):
        actual = circlify([self.data[1]])
        expected = [Element(1, 0.05, 'a2', None, Circle(0, 0, 1))]
        self.assertEqual(expected, actual)


class HedgeTestCase(unittest.TestCase):

    def test_one_big_two_small(self):
        """Makes sure we get 3 circles in t"""
        actual = circ.circlify([0.998997995991984, 0.000501002004008016,
                                0.000501002004008016])
        self.assertEqual(3, len(actual))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level='INFO')
    unittest.main()
