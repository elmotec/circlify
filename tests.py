#!/usr/bin/env python
# encoding: utf-8

"""Tests for circlify"""

import unittest

import circlify as circ
import hypothesis as h
import hypothesis.strategies as hst

# Set this variable to True to get a display of the layout (req matlplotlib)
display_layout = False


class CircleTest(unittest.TestCase):
    """Check behaviour of Circle class"""

    def test_repr_and_str_with_id_(self):
        """Check str() and repr() conversion"""
        expected = "Circle(x=0.0, y=0.0, r=2.0, level=1, " "ex={'label': '3'})"
        actual = str(circ.Circle(r=2.0, ex={"label": "3"}))
        self.assertEqual(expected, actual)

    def test_repr_and_str_with_circle(self):
        """Check str() and repr() conversion"""
        expected = "Circle(x=0.0, y=0.0, r=1.0, level=1, ex=None)"
        self.assertEqual(expected, str(circ.Circle()))

    def test_equality_with_ex(self):
        """Check equality with extended info."""
        self.assertNotEqual(circ.Circle(), circ.Circle(ex={"test": 0}))

    def test_unpack(self):
        """Circle should unpack to tuples."""
        x, y, r = circ.Circle(1, 2, 3)
        self.assertEqual((x, y, r), (1, 2, 3))


class SpecialCases(unittest.TestCase):
    """Hedge cases and obvious answers handling."""

    def test_empty_input(self):
        """What do we do when the output is empty?"""
        actual = circ.circlify([])
        self.assertEqual(actual, [])

    def test_single_value(self):
        """If there is only one value, it should occupy the whole circle."""
        actual = circ.circlify([2.0])
        expected = [circ.Circle(r=1.0, ex={"datum": 2.0})]
        self.assertEqual(actual, expected)

    def test_two_equal_values(self):
        """Two equal circle cases is also trivial."""
        actual = circ.circlify([1.0, 1.0])
        expected = [
            circ.Circle(x=0.5, r=0.5, ex={"datum": 1.0}),
            circ.Circle(x=-0.5, r=0.5, ex={"datum": 1.0}),
        ]
        self.assertEqual(actual, expected)

    def test_negative_values(self):
        """A ValueError exception should be thrown when there are negative values."""
        with self.assertRaises(ValueError) as context:
            _ = circ.circlify([-1.0] * 3)
        self.assertIn("must be positive", str(context.exception))

    def test_tiny_values_warning(self):
        """Tiny values cause stability issues and should generate a warning."""
        with self.assertLogs(circ.__name__, level="WARNING") as context:
            try:
                _ = circ.circlify([5e-324, 5e-324, 5e-324])
            except ValueError:
                pass
        self.assertIn("is small", context.output[0])

    def test_low_min_max_ratio_warning(self):
        """Low min to max ratio in the data generates should generate a warning."""
        with self.assertLogs(circ.log, level="WARNING") as context:
            try:
                _ = circ.circlify([1.0, 1.0, 2.9514790517935283e20])
            except ValueError:
                pass
        self.assertIn("min to max ratio", context.output[0])


class DisplayedTestCase(unittest.TestCase):
    """Display the result of the placement of the circle."""

    def display(self, circles, labels=None):
        """Forwards call to circ.circlify.bubbles()."""
        try:
            if display_layout:
                circ.bubbles(circles, labels)
        except AttributeError as err:
            print("{}. Did you install matplotlib?".format(err))
            raise


def density(circles):
    """Shortcut to compute density for a configuration output."""
    return circ.density([c.circle for c in circles])


class DensityThresholdTestCase(DisplayedTestCase):
    """Simple test cases that checks the density of the circle placement."""

    def setUp(self):
        self.density_threshold = 0.63

    def test_prime_series(self):
        """Check the coordinates of the circles returned are expected."""
        data = [19, 17, 13, 11, 7, 5, 3, 2, 1]
        actual = circ.circlify(data, show_enclosure=True)
        self.display(actual, reversed(data + [None]))
        self.assertGreater(density(actual), self.density_threshold)

    def test_range_series(self):
        """Check the coordinates of the circles returned are expected."""
        data = list(range(7, 1, -1))
        actual = circ.circlify(data, show_enclosure=True)
        self.display(actual, reversed(data + [None]))
        self.assertGreater(density(actual), self.density_threshold)

    def test_geometric_series(self):
        """Check the coordinates of the circles returned are expected."""
        data = sorted((2**n for n in range(4, 12)), reverse=True)
        actual = circ.circlify(data, show_enclosure=True)
        self.display(actual, reversed(data + [None]))
        self.assertGreater(density(actual), self.density_threshold)

    def test_many_similar_circles(self):
        """Check that many similar circle are packed as expected."""
        pi = 1.0
        data = (
            [{"id": "2.4", "datum": pi * 2.4**2}] * 1
            + [{"id": "1.825", "datum": pi * 1.825**2}] * 1
            + [{"id": "1.55", "datum": pi * 1.55**2}] * 6
            + [{"id": "1.275", "datum": pi * 1.275**2}] * 8
            + [{"id": "1.1875", "datum": pi * 1.1875**2}] * 9
        )
        actual = circ.circlify(data, show_enclosure=True)
        self.display(actual, range(len(actual)))
        self.assertGreater(density(actual), self.density_threshold)


class MultiInstanceTestCase(unittest.TestCase):
    """Multiple instances test cases."""

    @h.given(
        hst.lists(hst.floats(min_value=0.1, max_value=100), min_size=3, max_size=30)
    )
    def test_hypothesis(self, data):
        actual = circ.circlify(data, show_enclosure=True)
        self.assertGreaterEqual(density(actual), 0.5, str(data))

    def test_output_performance(self):
        """Test output peformance vs paper examples.

        See README.

        """
        # fmt: off
        instances = {
            "NR10-1*": [10, 12, 15, 20, 21, 30, 30, 30, 40, 50, 99.89],
            "NR11-1*": [8.4, 11, 10, 10.5, 12, 14, 15, 20, 20, 25, 25, 60.71],
            "NR12-1": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 65.30],
            "NR14-1": [11, 14, 15, 16, 17, 19, 23, 27, 31, 35, 36, 37, 38, 40, 113.84],
            "NR15-1": [3, 3, 4, 4, 4.5, 6, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 38.97],
            "NR15-2*": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 38.85],
            "NR16-1": [13, 14, 15, 15, 17, 19, 23, 26, 27, 27, 32, 37, 38, 47, 57, 63, 143.44, ],
            "NR16-2": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 128.29, ],
            "NR17-1*": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 15, 15, 20, 25, 49.25, ],
            "NR18-1": [12, 14, 16, 23, 25, 26, 27, 28, 33, 35, 47, 49, 53, 53, 55, 60, 67, 71, 197.40, ],
            "NR20-1": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 125.53, ],
            "NR20-2": [6, 8, 9, 12, 12, 15, 16, 18, 20, 21, 24, 24, 27, 28, 30, 32, 33, 36, 40, 44, 122.21, ],
            "NR21-1": [10, 15, 16, 17, 17, 18, 21, 22, 23, 25, 26, 31, 33, 34, 37, 37, 38, 39, 40, 42, 45, 148.82, ],
            "NR23-1": [14, 14, 16, 18, 18, 21, 22, 23, 26, 28, 28, 32, 34, 34, 36, 37, 39, 41, 45, 48, 49, 49, 51, 175.47, ],
            "NR24-1": [9, 10, 11, 13, 13, 16, 17, 17, 18, 19, 19, 20, 20, 20, 21, 22, 23, 23, 24, 25, 30, 31, 32, 82, 138.38, ],
            "NR25-1": [14, 17, 22, 26, 26, 27, 28, 29, 29, 30, 31, 32, 33, 34, 34, 34, 34, 35, 37, 37, 37, 47, 52, 53, 55, 190.47, ],
            "NR26-1": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 246.75, ],
            "NR26-2": [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 303.38, ],
            "NR27-1": [17, 21, 25, 26, 26, 27, 27, 28, 29, 33, 33, 34, 35, 35, 35, 37, 40, 42, 43, 44, 45, 49, 53, 55, 55, 55, 63, 222.58, ],
            "NR30-1": [5, 8, 10, 10, 12, 14, 15, 16, 18, 20, 20, 20, 20, 20, 22, 24, 25, 26, 30, 30, 30, 30, 35, 40, 40, 45, 48, 50, 55, 60, 178.66, ],
            "NR30-2": [6, 8, 8, 10, 12, 13, 14, 16, 18, 18, 20, 22, 23, 24, 25, 27, 28, 29, 31, 33, 33, 35, 37, 38, 39, 41, 43, 43, 48, 53, 173.70, ],
            "NR40-1": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 357.00, ],
            "NR50-1": [18, 18, 19, 19, 19, 19, 21, 21, 24, 25, 25, 30, 31, 31, 33, 33, 36, 36, 40, 42, 43, 46, 46, 47, 49, 49, 49, 50, 50, 54, 56, 56, 57, 57, 58, 58, 59, 59, 59, 61, 62, 63, 63, 64, 65, 68, 78, 79, 80, 86, 380.00, ],
            "NR60-1": [35, 35, 35, 36, 37, 37, 38, 38, 39, 39, 40, 41, 41, 42, 42, 42, 42, 42, 44, 44, 45, 45, 46, 46, 47, 48, 48, 49, 50, 50, 54, 54, 57, 57, 59, 60, 60, 71, 71, 71, 72, 72, 74, 74, 76, 77, 77, 79, 79, 80, 82, 82, 85, 86, 89, 90, 94, 95, 96, 100, 522.93, ],
        }
        # fmt: on
        actual_d, target_d = [], []
        for (name, data) in instances.items():
            data, target_r = data[:-1], data[-1]
            solution = circ.circlify(data)
            actual_d.append(circ.density([c.circle for c in solution]))
            target_d.append(sum(d**2.0 for d in data) / target_r**2.0)
        ratios = [a / t for a, t in zip(actual_d, target_d)]
        self.assertGreater(min(ratios), 0.76)
        self.assertGreater(sum(actual_d), sum(target_d) * 0.83)


class EnclosureScalingTestCase(unittest.TestCase):
    """Test circ.circ.scale function"""

    def test_simple_zoom(self):
        """Trivial zoom test when the enclosure is the same as the circle."""
        input_ = circ._Circle(0, 0, 0.5)
        target = circ._Circle(0, 0, 1.0)
        actual = circ.scale(input_, target, input_)
        self.assertEqual(target, actual)

    def test_simple_zoom_off_center(self):
        """Zoom test with off center circle equal to enclosure."""
        input_ = circ._Circle(0.5, 0.5, 0.5)
        target = circ._Circle(0.5, 0.5, 1.0)
        actual = circ.scale(input_, target, input_)
        self.assertEqual(target, actual)

    def test_simple_zoom_and_translation(self):
        """Pan and zoom test with off center circle equal to enclosure."""
        input_ = circ._Circle(0.5, 0.5, 0.5)
        target = circ._Circle(-0.5, 0, 1.0)
        actual = circ.scale(input_, target, input_)
        self.assertEqual(target, actual)

    def test_zoom_with_enclosure(self):
        """Zoom test with off center circle and difference enclosure"""
        input_ = circ._Circle(1.0, 0.0, 1.0)
        enclosure = circ._Circle(0.0, 0.0, 2.0)
        target = circ._Circle(0.0, 0.0, 1.0)
        actual = circ.scale(input_, target, enclosure)
        expected = circ._Circle(0.5, 0.0, 0.5)
        self.assertEqual(expected, actual)


class HandleDataTestCase(unittest.TestCase):
    """Test circlify._handle function."""

    def test_integer(self):
        """handles integer"""
        actual = circ._handle([42], 1)
        self.assertEqual([circ.Circle(r=42, ex={"datum": 42})], actual)

    def test_float(self):
        """Handles float."""
        actual = circ._handle([42.0], 1)
        self.assertEqual([circ.Circle(r=42.0, ex={"datum": 42.0})], actual)

    def test_dict_w_datum_only(self):
        """Handles dict with just the data"""
        actual = circ._handle([{"datum": 42}], 1)
        self.assertEqual([circ.Circle(r=42, ex={"datum": 42})], actual)

    def test_dict_w_datum_and_id(self):
        """Handles dict with data and an id"""
        actual = circ._handle([{"datum": 1, "id": "42"}], 1)
        self.assertEqual([circ.Circle(r=1, ex={"id": "42", "datum": 1})], actual)

    def test_bad_value_raise_error(self):
        """A set of non-dict, non-numeric input raises ValueError."""
        with self.assertRaises(TypeError):
            circ._handle({"datum", 42}, 1)

    def test_bad_dict_keys_raise_error(self):
        """A dict with the wrong key raises ValueError."""
        with self.assertRaises(TypeError):
            circ._handle({"datatum": 42}, 1)

    def test_handle_children(self):
        """A dict that has children."""
        actual = circ._handle([{"datum": 42, "children": [1, 2]}], 1)
        expected = [
            circ.Circle(r=42.0, ex={"datum": 42, "children": [1, 2]}),
        ]
        self.assertEqual(expected, actual)


def ignore_xyr(circles):
    """Change all x and y to 0.0 and r to 1.0

    This is useful for those tests whose actual (x, y, r) data can change.

    """
    return [circ.Circle(level=c.level, ex=c.ex) for c in circles]


class MultiLevelInputTestCase(DisplayedTestCase):
    """Handles multi-layer input."""

    def setUp(self):
        """Sets up the test case."""
        self.data = [
            0.05,
            {"id": "a2", "datum": 0.05},
            {
                "id": "a0",
                "datum": 0.8,
                "children": [0.3, 0.2, 0.2, 0.1],
            },
            {
                "id": "a1",
                "datum": 0.1,
                "children": [
                    {"id": "a1_1", "datum": 0.05},
                    {"datum": 0.04},
                    {"id": "a1_2", "datum": 0.01},
                ],
            },
        ]

    def test_json_input(self):
        """Simple json data."""
        actual = circ.circlify(self.data, show_enclosure=True)
        self.display(actual)
        expected = [
            circ.Circle(x=0.0, y=0.0, r=1.0, level=0, ex=None),
            circ.Circle(
                level=1,
                ex={"datum": 0.05},
            ),
            circ.Circle(
                level=1,
                ex={"id": "a2", "datum": 0.05},
            ),
            circ.Circle(
                level=1,
                ex={
                    "id": "a1",
                    "datum": 0.1,
                    "children": [
                        {"id": "a1_1", "datum": 0.05},
                        {"datum": 0.04},
                        {"id": "a1_2", "datum": 0.01},
                    ],
                },
            ),
            circ.Circle(
                level=1,
                ex={"id": "a0", "datum": 0.8, "children": [0.3, 0.2, 0.2, 0.1]},
            ),
            circ.Circle(
                level=2,
                ex={"id": "a1_2", "datum": 0.01},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.04},
            ),
            circ.Circle(
                level=2,
                ex={"id": "a1_1", "datum": 0.05},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.1},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.2},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.2},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.3},
            ),
        ]
        actual_level_and_ex = ignore_xyr(actual)
        self.assertEqual(expected, actual_level_and_ex)

    def test_handle_single_value(self):
        """Typical specification of data with just a value."""
        actual = circ.circlify([self.data[0]])
        expected = [circ.Circle(ex={"datum": 0.05})]
        self.assertEqual(expected, actual)

    def test_handle_custom_datum_key(self):
        """Specify value as dict with custom keys."""
        actual = circ.circlify([{"value": 0.05}], datum_field="value")
        expected = [circ.Circle(ex={"value": 0.05})]
        self.assertEqual(expected, actual)

    def test_handle_custom_id_key(self):
        """Specify value as dict with custom keys."""
        actual = circ.circlify([{"name": "a2", "datum": 0.05}], id_field="name")
        expected = [circ.Circle(ex={"name": "a2", "datum": 0.05})]
        self.assertEqual(expected, actual)

    def test_handle_dict(self):
        """Specify value as a dict."""
        actual = circ.circlify([self.data[1]])
        expected = [circ.Circle(ex={"id": "a2", "datum": 0.05})]
        self.assertEqual(expected, actual)

    def test_handle_dict_w_children(self):
        actual = circ.circlify([self.data[2]])
        expected = [
            circ.Circle(
                level=1,
                ex={"id": "a0", "datum": 0.8, "children": [0.3, 0.2, 0.2, 0.1]},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.1},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.2},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.2},
            ),
            circ.Circle(
                level=2,
                ex={"datum": 0.3},
            ),
        ]
        actual_level_and_ex = ignore_xyr(actual)
        self.assertEqual(expected, actual_level_and_ex)

    def test_missing_datum_value(self):
        """Missing data generates KeyError."""
        with self.assertRaises(KeyError):
            circ.circlify([{"ex": " Missing value"}])

    def test_missing_datum_value_w_datum_field(self):
        """Missing data generates KeyError with correct key name."""
        with self.assertRaisesRegex(KeyError, "value"):
            circ.circlify([{"ex": " Missing value"}], datum_field="value")


class HedgeTestCase(unittest.TestCase):
    def test_one_big_two_small(self):
        """Makes sure we get 3 circles in t"""
        actual = circ.circlify(
            [0.998997995991984, 0.000501002004008016, 0.000501002004008016]
        )
        self.assertEqual(3, len(actual))


class GetIntersectionTestCase(unittest.TestCase):
    """Test Circle.get_intersecton() edge cases."""

    def test_tiny_circle_contained_inside_other(self):
        """Testing for numerical problems with floating point math

        When the inner circle is right on the outer circle, we need to handle
        possible DomainError raised by the sqrt of negative number.

        """
        c1 = circ._Circle(
            x=-0.005574001032652584, y=0.10484176298731643, r=0.05662982038967889
        )
        c2 = circ._Circle(
            x=0.029345054623395653, y=0.06025929883988402, r=2.220446049250313e-15
        )
        self.assertEqual(circ.get_intersection(c1, c2), (None, None))

    def test_small_circle_that_does_intersect(self):
        """Testing for small circle that can be computed

        At the same time, the condition should not throw out configuration
        that can be computed.

        """
        c1 = circ._Circle(
            x=-0.005574001032652584, y=0.10484176298731643, r=0.05662982038967889
        )
        c2 = circ._Circle(x=0.029345054623395653, y=0.06025929883988402, r=1.0e-09)
        self.assertEqual(
            circ.get_intersection(c1, c2),
            (
                (0.029345053725419824, 0.06025929813654767),
                (0.029345055521371476, 0.06025929954322038),
            ),
        )

    def test_degenerate_inner_completely_inside_outer_circle(self):
        """Testing case where degenerate inner is completely inside the outer circle."""
        c1 = circ.Circle(0, 0, 0)
        c2 = circ.Circle(0, 0, 1)
        self.assertEqual(circ.get_intersection(c1, c2), (None, None))

    @h.given(
        hst.floats(),
        hst.floats(),
        hst.floats(),
        hst.floats(),
        hst.floats(),
        hst.floats(),
    )
    def test_edge_cases(self, x1, y1, r1, x2, y2, r2):
        """Edge cases do not cause exceptions."""
        c1 = circ.Circle(x=x1, y=y1, r=r1)
        c2 = circ.Circle(x=x2, y=y2, r=r2)
        self.assertIsNotNone(circ.get_intersection(c1, c2))


class LookAheadLoopTestCase(unittest.TestCase):
    """Test look ahead loop."""

    def test_one_look_ahead(self):
        """Normal case of a look ahead."""
        actual = list(circ.look_ahead([1, 2, 3]))
        expected = [(1, 2), (2, 3), (3, None)]
        self.assertEqual(actual, expected)

    def test_one_look_ahead_with_empty_input(self):
        """Empty input case of a look ahead."""
        actual = list(circ.look_ahead([]))
        self.assertEqual(actual, [])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level="INFO")
    circ.log.setLevel(logging.INFO)
    unittest.main()
