circlify
========

Pure Python implementation of circle packing layout algorithm.

Circles are first arranged via a version of A1.0 by Huang et al (see https://home.mis.u-picardie.fr/~cli/Publis/circle.pdf for details) and then enclosed in a circle created around them using Matoušek-Sharir-Welzl algorithm used in d3js (see https://beta.observablehq.com/@mbostock/miniball, https://beta.observablehq.com/@mbostock/miniball, and https://github.com/d3/d3-hierarchy/blob/master/src/pack/enclose.js)

Installation
------------

Compatible with Python 2.7 and Python 3.2+.

Using pip:

::

    pip install circlify

or using the source:

:: 

    git clone git://github.com/elmotec/circlify.git
    cd circlify
    python setup.py install


The last step may require `sudo` if you don't have root access.  The `setup.py`
script uses `setuptools`/`distribute`.


Usage
-----

The main function `circlify` is supported by a small data class `circlify.Circle` and takes 3 parameters:

* A list of positive values sorted from largest to smallest.
* (optional) A target enclosure where the packed circles should fit. It defaults to the unit circle (0, 0, 1).
* (optional) A boolean indicating if the target enclosure should be appended to the output.

The function returns a list of `circlify.Circle` objects, each one corresponding
to the coordinates and radius of cirlces proportional to the corresponding input value.


Example
-------

.. code:: python

  import circlify as circ

  data = [19, 17, 13, 11, 7, 5, 3, 2, 1]
  circles = circ.circlify(self.data, with_enclosure=True)


The variable `circles` contains (last one is the enclosure):

.. code:: python

  [circ.Circle(x=0.35776879346704843, y=-0.13064957525245907, r=0.39529216048201216),
   circ.Circle(x=-0.411432317820337, y=-0.13064957525245907, r=0.3739089508053733),
   circ.Circle(x=-0.04661299415374866, y=0.4678014425767657, r=0.32697389223002427),
   circ.Circle(x=-0.045884607890591435, y=-0.6977206243364218, r=0.3007722353441051),
   circ.Circle(x=-0.6132109517981927, y=0.4490810687795324, r=0.23993324126007678),
   circ.Circle(x=0.48296614887228806, y=0.4541723195782383, r=0.20278059970175755),
   circ.Circle(x=0.3252787490004198, y=0.7776370388468007, r=0.15707317711577193),
   circ.Circle(x=-0.40283175658099674, y=0.7512387781681531, r=0.12824971207048294),
   circ.Circle(x=0.09222041925800777, y=0.8617116738294696, r=0.09068624109026069),
   circ.Circle(x=0.0, y=0.0, r=1.0)]


A simple matplotlib representation. See `circlify.bubbles` helper function:

.. figure:: https://github.com/elmotec/circlify/blob/master/static/Figure_3.png
   :alt: visualization of circlify circle packing of first 9 prime numbers.

