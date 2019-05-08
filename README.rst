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
  circles = circ.circlify(data, with_enclosure=True)


The variable `circles` contains (last one is the enclosure):

.. code:: python

TODO


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
  circles = circ.circlify(data, with_enclosure=True)


returns:

.. code:: python

TODO


A simple matplotlib representation. See ``circlify.bubbles`` helper function (requires ``matplotlib``):

.. figure:: https://github.com/elmotec/circlify/blob/master/static/Figure_4.png
   :alt: visualization of circlify nested circle packing for a hierarchical input.

