[![PyPi version](https://img.shields.io/pypi/v/circlify.svg)](https://pypi.org/pypi/circlify/)
[![Python compatibility](https://img.shields.io/pypi/pyversions/circlify.svg)](https://pypi.org/pypi/circlify/)
[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/elmotec/circlify/Python%20package/main)](https://github.com/elmotec/circlify/actions)
[![Codecov](https://codecov.io/gh/elmotec/circlify/branch/master/graph/badge.svg?token=PSE4TFPGTV)](https://codecov.io/gh/elmotec/circlify)

# circlify

Pure Python implementation of a circle packing layout algorithm, inspired by [d3js](https://observablehq.com/@d3/zoomable-circle-packing) and [squarify](https://github.com/laserson/squarify).

Circles are first arranged with a euristic inspired by the A1.0 of [\[Huang-2006\]](#Huang-2006), then enclosed in a circle created around them using [\[MSW-1996\]](#MSW-1996) algorithm used in [\[Bostock-2017\]](#Bostock-2017). I hope to implement A1.5 at some point in the future but the results are good enough for my use case.

## Installation

Using pip:

```python
    pip install circlify
```

or using the source:

```python
    git clone git://github.com/elmotec/circlify.git
    cd circlify
    python setup.py install
```

The last step may require `sudo` if you don\\'t have root access.

## Usage

The main function `circlify` is supported by a small data class `circlify.Circle` and takes 3 parameters:

-   A list of positive values sorted from largest to smallest.

-   (optional) A target enclosure where the packed circles should fit.
    It defaults to the unit circle (0, 0, 1).

-   (optional) A boolean indicating if the target enclosure should be
    appended to the output.

The function returns a list of `circlify.Circle` whose _area_ is proportional to the corresponding input value.

### Example

```python
>>> from pprint import pprint as pp
>>> import circlify as circ
>>> circles = circ.circlify([19, 17, 13, 11, 7, 5, 3, 2, 1], show_enclosure=True)
>>> pp(circles)
[Circle(x=0.0, y=0.0, r=1.0, level=0, ex=None),
 Circle(x=-0.633232604611031, y=-0.47732413442115296, r=0.09460444572843042, level=1, ex={'datum': 1}),
 Circle(x=-0.7720311587589236, y=0.19946176418549022, r=0.13379089020993573, level=1, ex={'datum': 2}),
 Circle(x=-0.43168871955473165, y=-0.6391381648617572, r=0.16385970662353394, level=1, ex={'datum': 3}),
 Circle(x=0.595447603036083, y=0.5168251295666467, r=0.21154197162246005, level=1, ex={'datum': 5}),
 Circle(x=-0.5480911056188739, y=0.5115139053491098, r=0.2502998363185337, level=1, ex={'datum': 7}),
 Circle(x=0.043747233552068686, y=-0.6848366902134195, r=0.31376744998074435, level=1, ex={'datum': 11}),
 Circle(x=0.04298737651230445, y=0.5310431146935967, r=0.34110117996070605, level=1, ex={'datum': 13}),
 Circle(x=-0.3375943908160698, y=-0.09326467617622711, r=0.39006412239133215, level=1, ex={'datum': 17}),
 Circle(x=0.46484095011516874, y=-0.09326467617622711, r=0.4123712185399064, level=1, ex={'datum': 19})]
```

A simple matplotlib representation. See `circlify.bubbles` helper function (requires [matplotlib](https://matplotlib.org)):

![](https://github.com/elmotec/circlify/blob/main/static/Figure_3.png)

## Hierarchical circle packing

Starting with version 0.10, circlify also handle hierarchical input so that:

```python
>>> from pprint import pprint as pp
>>> import circlify as circ
>>> data = [
        0.05, {'id': 'a2', 'datum': 0.05},
        {'id': 'a0', 'datum': 0.8, 'children': [0.3, 0.2, 0.2, 0.1], },
        {'id': 'a1', 'datum': 0.1, 'children': [
            {'id': 'a1_1', 'datum': 0.05}, {'datum': 0.04}, 0.01],
        },
    ]
>>> circles = circ.circlify(data, show_enclosure=True)
>>> pp(circles)
[Circle(x=0.0, y=0.0, r=1.0, level=0, ex=None),
 Circle(x=-0.5658030759977484, y=0.4109778665114514, r=0.18469903125906464, level=1, ex={'datum': 0.05}),
 Circle(x=-0.5658030759977484, y=-0.4109778665114514, r=0.18469903125906464, level=1, ex={'id': 'a2', 'datum': 0.05}),
 Circle(x=-0.7387961250362587, y=0.0, r=0.2612038749637415, level=1, ex={'id': 'a1', 'datum': 0.1, 'children': [{'id': 'a1_1', 'datum': 0.05}, {'datum': 0.04}, 0.01]}),
 Circle(x=0.2612038749637414, y=0.0, r=0.7387961250362586, level=1, ex={'id': 'a0', 'datum': 0.8, 'children': [0.3, 0.2, 0.2, 0.1]}),
 Circle(x=-0.7567888163564135, y=0.1408782365133844, r=0.0616618704777984, level=2, ex={'datum': 0.01}),
 Circle(x=-0.8766762590444033, y=0.0, r=0.1233237409555968, level=2, ex={'datum': 0.04}),
 Circle(x=-0.6154723840806618, y=0.0, r=0.13788013400814464, level=2, ex={'id': 'a1_1', 'datum': 0.05}),
 Circle(x=0.6664952237042414, y=0.33692908734605553, r=0.21174557028487648, level=2, ex={'datum': 0.1}),
 Circle(x=-0.1128831469183017, y=-0.23039288135707192, r=0.29945345726929773, level=2, ex={'datum': 0.2}),
 Circle(x=0.1563193680487183, y=0.304601976765483, r=0.29945345726929773, level=2, ex={'datum': 0.2}),
 Circle(x=0.5533243963620487, y=-0.23039288135707192, r=0.3667540860110527, level=2, ex={'datum': 0.3})]
```

A simple matplotlib representation. See `circlify.bubbles` helper function (requires [matplotlib](https://matplotlib.org)):

![](https://github.com/elmotec/circlify/blob/main/static/Figure_4.png)

### Relative size of circles in hierachy

The area of the circles are proportional to the values passed in input only if the circles are at the same hierarchical level. For instance: circles _a1_1_ and _a2_ both have a value of 0.05, yet _a1_1_ is smaller than _a2_ because _a1_1_ is fitted within its parent circle _a1_ one level below the level of _a2_. In other words, the level 1 circles _a1_ and _a2_ are both proportional to their respective values but _a1_1_ is proportional to the values on level 2 witin _a1_.

### Invalid input

A warning is issued if a key is not understood. The check is disabled if the program is running with [-O or -OO option](https://docs.python.org/3/using/cmdline.html#cmdoption-O). One can also disable the warning with the [regular logging filters](https://docs.python.org/3/library/logging.html#filter-objects).

For instance:

```python
>>> import logging
>>> import sys
>>> import circlify as circ
>>> data = [ 0.05, {'id': 'a2', 'datum': 0.05, "bogus": {}}]
>>> logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
>>> _ = circ.circlify(data)
unexpected 'bogus' in input is ignored  # not issued if __debug__ is false
```

## References

### Bostock-2017

Mike Bostock, D3.js, <https://beta.observablehq.com/@mbostock/miniball>

### Huang-2006

WenQi HUANG, Yu LI, ChuMin LI, RuChu XU, New Heuristics for Packing Unequal Circles into a Circular Container, <https://home.mis.u-picardie.fr/~cli/Publis/circle.pdf>

### MSW-1996

J. Matoušek, M. Sharir, and E. Welzl. A Subexponential Bound For Linear Programming. Algorithmica, 16(4/5):498--516, October/November 1996, <http://www.inf.ethz.ch/personal/emo/PublFiles/SubexLinProg_ALG16_96.pdf>
