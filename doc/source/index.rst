.. PyMaxflow documentation master file, created by
   sphinx-quickstart on Wed Apr 20 07:08:53 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyMaxflow's documentation!
=====================================

PyMaxflow is a Python library for graph construction and
maxflow computation (commonly known as `graph cuts`)
as described in [BOYKOV04]_. The
core of this library is the C++ implementation by
Vladimir Kolmogorov, which
can be downloaded from his `homepage <http://pub.ist.ac.at/~vnk/software.html>`_.
Besides the wrapper to the C++ library, PyMaxflow offers

* NumPy integration.
* Methods for fast construction of common graph
  layouts in computer vision and graphics. This is
  probably the most powerful feature of PyMaxflow,
  since the creation of large graphs
  using the standard "one edge per call" functions
  of the C++ library is extremely slow from Python.
* Implementation of algorithms for fast energy
  minimization which use the `maxflow` method: the :math:`\alpha\beta`-swap
  and the :math:`\alpha`-expansion.

Take a look at the :ref:`tutorial`.

Contents
========

.. toctree::
   :maxdepth: 2
   
   tutorial
   maxflow

License
=======

This software is licensed under the GPL.

.. important::
   The core of the library is
   the C++ implementation by Vladimir Kolmogorov. It is also
   licensed under the GPL, but it **REQUIRES** that you cite [BOYKOV04]_
   in any resulting publication if you use this code for
   research purposes. This requirement extends to *PyMaxflow*.

Bugs and wishes on the version |version|
========================================

*PyMaxflow* has been used
under different conditions but it has not been thoroughly
tested. Therefore, you might find bugs.

If you find bugs or have some special needs that you
think should be available in *PyMaxflow*, please
let me know.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [BOYKOV04] *An Experimental Comparison of Min-Cut/Max-Flow Algorithms for
   Energy Minimization in Vision.* Yuri Boykov and Vladimir Kolmogorov. In
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), September 2004
