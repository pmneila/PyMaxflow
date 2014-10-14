PyMaxflow is a Python library for graph construction and maxflow computation
(commonly known as `graph cuts`). The core of this library is the C++
implementation by Vladimir Kolmogorov, which can be downloaded from his
`homepage <http://pub.ist.ac.at/~vnk/software.html>`_. Besides the wrapper to
the C++ library, PyMaxflow offers

* NumPy integration, 
* methods for fast construction of common graph
  layouts in computer vision and graphics,
* implementation of algorithms for fast energy
  minimization which use the ``maxflow`` method: the αβ-swap
  and the α-expansion.

Take a look at the `PyMaxflow documentation <http://pmneila.github.com/PyMaxflow/>`_.

Requirements
------------

You need the following libraries and tools installed on you system in order to
build PyMaxflow:

* `NumPy <http://numpy.scipy.org/>`_
* `Cython <http://cython.org/>`_

They should be automatically downloaded and installed if PyMaxflow is installed
via PyPI using ``pip install`` (see below).

Installation via PyPI
---------------------

Open a terminal and write::

  $ pip install PyMaxflow

Depending on your local configuration, you might need admin privileges. In that
case, login as root or use ``sudo`` before ``pip install``.

Manual installation
-------------------

Download the source code or clone the Github repository. Then, open a terminal
and write::

  $ cd path/to/PyMaxflow
  $ python setup.py build
  ... lots of text ...

If everything went OK, you should be able to install the
package with::

  $ sudo python setup.py install


Documentation
-------------

The documentation of the package is available under the ``doc``
directory. To generate the HTML documentation, use::

  $ cd doc/
  $ make html

