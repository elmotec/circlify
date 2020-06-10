.. image:: https://img.shields.io/pypi/v/circlify.svg
    :target: https://pypi.org/pypi/circlify/
    :alt: PyPi version

.. image:: https://img.shields.io/pypi/pyversions/circlify.svg
    :target: https://pypi.org/pypi/circlify/
    :alt: Python compatibility

.. image:: https://img.shields.io/travis/elmotec/circlify.svg
    :target: https://travis-ci.org/elmotec/circlify
    :alt: Build Status

.. image:: https://coveralls.io/repos/elmotec/circlify/badge.svg
    :target: https://coveralls.io/r/elmotec/circlify
    :alt: Coverage

.. image:: https://img.shields.io/codacy/grade/474b0af6853a4c5f8f9214d3220571f9.svg
    :target: https://www.codacy.com/app/elmotec/circlify/dashboard
    :alt: Codacy


========
circlify
========

Pure Python implementation of circle packing layout algorithm.

Circles are first arranged via a version of A1.0 by Huang et al (see https://home.mis.u-picardie.fr/~cli/Publis/circle.pdf for details) and then enclosed in a circle created around them using Matoušek-Sharir-Welzl algorithm used in d3js (see https://beta.observablehq.com/@mbostock/miniball, http://www.inf.ethz.ch/personal/emo/PublFiles/SubexLinProg_ALG16_96.pdf, and https://github.com/d3/d3-hierarchy/blob/master/src/pack/enclose.js)

Installation
------------

Using pip:

::

    pip install circlify

or using the source:

:: 

    git clone git://github.com/elmotec/circlify.git
    cd circlify
    python setup.py install


The last step may require ``sudo`` if you don't have root access.


Usage
-----

The main function ``circlify`` is supported by a small data class ``circlify.Circle`` and takes 3 parameters:

* A list of positive values sorted from largest to smallest.
* (optional) A target enclosure where the packed circles should fit. It defaults to the unit circle (0, 0, 1).
* (optional) A boolean indicating if the target enclosure should be appended to the output.

The function returns a list of ``circlify.Circle`` objects, each one corresponding
to the coordinates and radius of cirlces proportional to the corresponding input value.


Example
-------

.. code:: python

  import circlify as circ

  data = [19, 17, 13, 11, 7, 5, 3, 2, 1]
  circles = circ.circlify(data, show_enclosure=True)


The variable `circles` contains (level 0 is the enclosure):

.. code:: python

    [
        circ.Circle(x=0.0, y=0.0, r=1.0, level=0),
        circ.Circle(x=0.09222041925800777, y=0.8617116738294696,
                    r=0.09068624109026069),
        circ.Circle(x=-0.40283175658099674, y=0.7512387781681531,
                    r=0.12824971207048294),
        circ.Circle(x=0.3252787490004198, y=0.7776370388468007,
                    r=0.15707317711577193),
        circ.Circle(x=0.48296614887228806, y=0.4541723195782383,
                    r=0.20278059970175755),
        circ.Circle(x=-0.6132109517981927, y=0.4490810687795324,
                    r=0.23993324126007678),
        circ.Circle(x=-0.045884607890591435, y=-0.6977206243364218,
                    r=0.3007722353441051),
        circ.Circle(x=-0.04661299415374866, y=0.4678014425767657,
                    r=0.32697389223002427),
        circ.Circle(x=-0.411432317820337, y=-0.13064957525245907,
                    r=0.3739089508053733),
        circ.Circle(x=0.35776879346704843, y=-0.13064957525245907,
                    r=0.39529216048201216),
    ]


A simple matplotlib representation. See ``circlify.bubbles`` helper function (requires ``matplotlib``):

.. figure:: https://github.com/elmotec/circlify/blob/master/static/Figure_3.png
   :alt: visualization of circlify circle packing of first 9 prime numbers.

Starting with version 0.10, circlify also handle hierarchical input so that:

.. code:: python

  import circlify as circ

  data = [0.05, {'id': 'a2', 'datum': 0.05},
          {'id': 'a0', 'datum': 0.8, 'children': [0.3, 0.2, 0.2, 0.1], },
          {'id': 'a1', 'datum': 0.1, 'children':
            [ {'id': 'a1_1', 'datum': 0.05}, {'datum': 0.04}, 0.01],},
        ]
  circles = circ.circlify(data, show_enclosure=True)


returns:

.. code:: python

    [
        circ.Circle(level=0, r=1.0),
        circ.Circle(x=-0.565803075997749, y=0.41097786651145324,
                    r=0.18469903125906464),
        circ.Circle(x=-0.3385727489559141, y=0.7022188441650276,
                    r=0.18469903125906464, ex={'id': 'a2', 'datum': 0.05}),
        circ.Circle(x=-0.7387961250362587, r=0.2612038749637415,
                    ex={'id': 'a1', 'datum': 0.1,
                        'children': [{'id': 'a1_1', 'datum': 0.05},
                                     {'datum': 0.04},
                                     {'id': 'a1_2', 'datum': 0.01}]}),
        circ.Circle(x=0.2612038749637414, r=0.7387961250362586,
                    ex={'id': 'a0', 'datum': 0.8,
                        'children': [0.3, 0.2, 0.2, 0.1]}),
        circ.Circle(level=2, x=-0.7567888163564136,
                    y=0.14087823651338607, r=0.0616618704777984,
                    ex={'id': 'a1_2', 'datum': 0.01}),
        circ.Circle(level=2, x=-0.8766762590444033, y=0.0,
                    r=0.1233237409555968,
                    ex={'datum': 0.04}),
        circ.Circle(level=2, ex={'id': 'a1_1', 'datum': 0.05},
                    x=-0.6154723840806618, y=0.0, r=0.13788013400814464),
        circ.Circle(level=2, x=0.6664952237042423,
                    y=0.3369290873460549, r=0.2117455702848763),
        circ.Circle(level=2, x=-0.11288314691830154,
                    y=-0.230392881357073, r=0.2994534572692975),
        circ.Circle(level=2, x=0.15631936804871832,
                    y=0.30460197676548245, r=0.2994534572692975),
        circ.Circle(level=2, x=0.5533243963620484,
                    y=-0.230392881357073, r=0.36675408601105247),
    ]


A simple matplotlib representation. See ``circlify.bubbles`` helper function (requires ``matplotlib``):

.. figure:: https://github.com/elmotec/circlify/blob/master/static/Figure_4.png
   :alt: visualization of circlify nested circle packing for a hierarchical input.

*Note* that the area of the circles are proportional to the values passed in input only if the circles are at the same hierarchical level.
For instance: circles *a1_1* and *a2* both have a value of 0.05, yet *a1_1* is smaller than *a2* because *a1_1* is fitted within its parent circle *a1* one level below the level of *a2*.
In other words, the level 1 circles *a1* and *a2* are both proportional to their respective values but *a1_1* is proportional to the values on level 2 witin *a1*.
