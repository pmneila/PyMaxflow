.. PyMaxflow documentation master file, created by
   sphinx-quickstart on Wed Apr 20 07:08:53 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyMaxflow's documentation!
=====================================

*PyMaxflow* is a Python library to build flow networks and compute their maximum
flow/minimum cut (commonly known as `graph cuts`) as described in [BOYKOV04]_.
This is a common technique used in different problems of image processing,
computer vision and computer graphics. The core of this library is the C++
maxflow implementation by Vladimir Kolmogorov, which can be downloaded from his
`homepage <http://pub.ist.ac.at/~vnk/software.html>`_. Besides being a wrapper
to the C++ library, PyMaxflow also offers

* NumPy integration,
* methods for fast declaration of complex network layouts with a single API
  call, which avoids the much slower one-call-per-edge alternative offered by
  the wrapped functions of the core C++ library, and
* implementation of algorithms for fast energy minimization with more than two
  labels: the αβ-swap and the α-expansion.

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
   The core of the library is the C++ implementation by Vladimir Kolmogorov. It
   is also licensed under the GPL, but it **REQUIRES** that you cite [BOYKOV04]_
   in any resulting publication if you use this code for research purposes. This
   requirement extends to *PyMaxflow*.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [BOYKOV04] *An Experimental Comparison of Min-Cut/Max-Flow Algorithms for
   Energy Minimization in Vision.* Yuri Boykov and Vladimir Kolmogorov. In
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), September 2004
