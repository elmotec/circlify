#!/usr/bin/env python
# encoding: utf-8

"""Tests for circlify"""


import unittest

from circlify import Circle, circlify


# Set this variable to True to get a display of the layout (req matlplotlib)
display_layout = False


class SpecialCases(unittest.TestCase):
    """Hedge cases and obvious answers handling."""

    def test_empty_input(self):
        """What do we do when the output is empty?"""
        actual = circlify([])
        self.assertEqual(actual, [])

    def test_single_value(self):
        """If there is only one value, it should occupy the whole circle."""
        actual = circlify([2.0])
        expected = [Circle(0.0, 0.0, 1.0)]
        self.assertEqual(actual, expected)

    def test_two_equal_values(self):
        """Two equal circle cases is also trivial."""
        # Force scaling to .5 so that each circle radius is brought to 0.5.
        actual = circlify([1.0, 1.0])
        expected = [Circle(0.5, 0.0, 0.5), Circle(-0.5, 0.0, 0.5)]
        self.assertEqual(actual, expected)


class TestCaseWithDisplay(unittest.TestCase):
    """Display the result of the placement of the circle."""

    def display(self, circles, labels):
        """Forwards call to circlify.bubbles()."""
        try:
            if display_layout:
                from circlify import bubbles
                bubbles(circles, labels)
        except AttributeError as err:
            log.error("%s. %s", err, "Did you install matplotlib?")


class PrimeSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of prime number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, 3, ... up to 19."""
        self.data = [19, 17, 13, 11, 7, 5, 3, 2, 1]

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circlify(self.data, with_target=True)
        expected = [Circle(x=0.35776879346704843, y=-0.13064957525245907,
                           r=0.39529216048201216),
                    Circle(x=-0.411432317820337, y=-0.13064957525245907,
                           r=0.3739089508053733),
                    Circle(x=-0.04661299415374866, y=0.4678014425767657,
                           r=0.32697389223002427),
                    Circle(x=-0.045884607890591435, y=-0.6977206243364218,
                           r=0.3007722353441051),
                    Circle(x=-0.6132109517981927, y=0.4490810687795324,
                           r=0.23993324126007678),
                    Circle(x=0.48296614887228806, y=0.4541723195782383,
                           r=0.20278059970175755),
                    Circle(x=0.3252787490004198, y=0.7776370388468007,
                           r=0.15707317711577193),
                    Circle(x=-0.40283175658099674, y=0.7512387781681531,
                           r=0.12824971207048294),
                    Circle(x=0.09222041925800777, y=0.8617116738294696,
                           r=0.09068624109026069),
                    Circle(x=0.0, y=0.0, r=1.0)]
        self.display(actual, [str(v) for v in self.data])
        self.assertEqual(actual, expected)


class CountSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, ..."""
        self.data = list(range(7, 1, -1))

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circlify(self.data, with_target=True)
        expected = [Circle(x=0.5824456027453089, y=-0.08515409741642607,
                           r=0.41136250504733196),
                    Circle(x=-0.20976457776763055, y=-0.08515409741642607,
                           r=0.3808476754656075),
                    Circle(x=0.15769153632817096, y=0.5438978793053209,
                           r=0.34766477137653345),
                    Circle(x=0.15910532107887837, y=-0.6704181394216174,
                           r=0.31096082487194077),
                    Circle(x=-0.4586184780594718, y=0.5154819840108337,
                           r=0.2692999739208646),
                    Circle(x=-0.7680630545906644, y=0.13661056172475666,
                           r=0.21988250795031175),
                    Circle(x=0.0, y=0.0, r=1.0)]
        self.display(actual, [str(v) for v in self.data])
        self.assertEqual(actual, expected)


class GeometricSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, ..."""
        self.data = sorted([2 ** n for n in range(4, 12)], reverse=True)

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circlify(self.data, with_target=True)
        self.display(actual, [str(v) for v in self.data])
        expected = [Circle(x=0.4142135623730951, y=0.0, r=0.5857864376269051),
                    Circle(x=-0.5857864376269051, y=0.0, r=0.4142135623730951),
                    Circle(x=-0.2218254069479773, y=0.6062444788590926,
                           r=0.29289321881345254),
                    Circle(x=-0.20710678118654763, y=-0.49258571550470814,
                           r=0.20710678118654754),
                    Circle(x=0.10281914590763144, y=-0.662720719883036,
                           r=0.14644660940672627),
                    Circle(x=-0.11312522101671703, y=-0.7886890904910677,
                           r=0.10355339059327377),
                    Circle(x=0.041837742530372556, y=-0.8737565926802316,
                           r=0.07322330470336313),
                    Circle(x=-0.18045635173699437, y=-0.22990093891844118,
                           r=0.051776695296636886),
                    Circle(x=0.0, y=0.0, r=1.0)]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level='INFO')
    unittest.main()
