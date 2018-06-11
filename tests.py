#!/usr/bin/env python
# encoding: utf-8

"""Tests for circlify"""


import unittest

import circlify as circ


# Set this variable to True to get a display of the layout (req matlplotlib)
display_layout = False


class SpecialCases(unittest.TestCase):
    """Hedge cases and obvious answers handling."""

    def notest_empty_input(self):
        """What do we do when the output is empty?"""
        actual = circ.circlify({})
        self.assertEqual(actual, {})

    def notest_single_value(self):
        """If there is only one value, it should occupy the whole circle."""
        actual = circ.circlify({'a': 2.0})
        expected = {'a': circ.Circle(0.0, 0.0, 1.0)}
        self.assertEqual(actual, expected)

    def notest_two_equal_values(self):
        """Two equal circle cases is also trivial."""
        # Force scaling to .5 so that each circle radius is brought to 0.5.
        actual = circ.circlify({'a': 1.0, 'b': 1.0})
        expected = {'a': circ.Circle(0.5, 0.0, 0.5),
                    'b': circ.Circle(-0.5, 0.0, 0.5)}
        self.assertEqual(actual, expected)

    def notest_bubbles(self):
        """Test display of bubbles"""
        circ.bubbles({'unit': circ.Circle(0, 0, 1)})
        self.assertTrue(True)


class TestCaseWithDisplay(unittest.TestCase):
    """Display the result of the placement of the circle."""

    def display(self, actual):
        """Forwards call to circlify.bubbles()."""
        try:
            if display_layout:
                circ.bubbles(actual)
        except AttributeError as err:
            circ.log.error("%s. %s", err, "Did you install matplotlib?")


class PrimeSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of prime number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, 3, ... up to 19."""
        self.data = {'1': 1, '2': 2, '3': 3, '5': 5, '7': 7,
                     '11': 11, '13': 13, '17': 17, '19': 19}

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circ.circlify(self.data, with_unit=True)
        expected = {'19': circ.Circle(x=0.21678789382906316,
                                      y=-0.028783847098809247,
                                      r=0.3951965225695486),
                    '17': circ.Circle(x=-0.5522271151372987,
                                      y=-0.028783847098809247,
                                      r=0.37381848639681314),
                    '13': circ.Circle(x=-0.18749605671356087,
                                      y=0.5695223800874969,
                                      r=0.32689478339962186),
                    '11': circ.Circle(x=-0.1867678466778845,
                                      y=-0.5957176976830932,
                                      r=0.3006994658040256),
                    '7': circ.Circle(x=0.35112628105018256,
                                     y=-0.6494844830024308,
                                     r=0.23987519124892365),
                    '5': circ.Circle(x=0.701442000461867,
                                     y=-0.3789671749769543,
                                     r=0.20273153848784445),
                    '3': circ.Circle(x=0.7690104270712262,
                                     y=-0.02560249723891397,
                                     r=0.1570351744628859),
                    '2': circ.Circle(x=0.6657386565217119,
                                     y=0.24030106540102816,
                                     r=0.12821868303433528),
                    '1': circ.Circle(x=0.49108987864091774,
                                     y=0.3722387959995559,
                                     r=0.09066430024838702),
                    '': circ.Circle(x=0.0, y=0.0, r=1.0)}
        self.display(actual)
        self.assertEqual(actual, expected)


class CountSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, ..."""
        self.data = {str(n + 1): n + 1 for n in range(6)}

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circ.circlify(self.data, with_unit=True)
        expected = {'6': circ.Circle(x=0.2822331001132116,
                                     y=-0.03461740008821662,
                                     r=0.4050787227018648),
                    '5': circ.Circle(x=-0.4926302125706389,
                                     y=-0.03461740008821662,
                                     r=0.3697845899819857),
                    '4': circ.Circle(x=-0.13791069480709892,
                                     y=0.5694656401496978,
                                     r=0.3307453920926431),
                    '3': circ.Circle(x=-0.13589235783967413,
                                     y=-0.5853994123218175,
                                     r=0.2864339117368737),
                    '2': circ.Circle(x=0.3780798126137221,
                                     y=-0.6663387226474294,
                                     r=0.23387230959491145),
                    '1': circ.Circle(x=0.697042106506303,
                                     y=-0.42621452644002644,
                                     r=0.16537269604632154),
                    '': circ.Circle(x=0.0, y=0.0, r=1.0)}
        self.display(actual)
        self.assertEqual(actual, expected)


class GeometricSerieTestCase(TestCaseWithDisplay):
    """Consider a simple sequence of number for radius to our circles."""

    def setUp(self):
        """Sets up the primes sequence 1, 2, ..."""
        self.data = {str(2 ** (n + 4)): 2 ** (n + 4) for n in range(8)}

    def test_circlify(self):
        """Check the coordinates of the circles returned are expected."""
        actual = circ.circlify(self.data, with_unit=True)
        expected = {'2048': circ.Circle(x=0.4139278881798894,
                                        y=0.023904601986979225,
                                        r=0.5853824333084535),
                    '1024': circ.Circle(x=-0.5853824333084535,
                                        y=0.023904601986979225,
                                        r=0.4139278881798894),
                    '512': circ.Circle(x=-0.2216724187314656,
                                       y=0.6297309670561919,
                                       r=0.29269121665422676),
                    '256': circ.Circle(x=-0.2069639440899447,
                                       y=-0.46834138773459605,
                                       r=0.2069639440899447),
                    '128': circ.Circle(x=0.1027482337521121,
                                       y=-0.6383590536563235,
                                       r=0.14634560832711338),
                    '64': circ.Circle(x=-0.11304720098265524,
                                      y=-0.7642405465859983,
                                      r=0.10348197204497235),
                    '32': circ.Circle(x=0.04180888793837325,
                                      y=-0.8492493795468621,
                                      r=0.07317280416355669),
                    '16': circ.Circle(x=-0.18033189486890924,
                                      y=-0.2058377791940801,
                                      r=0.051740986022486175),
                    '': circ.Circle(x=0.0, y=0.0, r=1.0)}
        self.display(actual)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level='INFO')
    unittest.main()
