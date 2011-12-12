PyMaxflow is a Python library for graph construction and
maxflow computation (commonly known as `graph cuts`)
as described in Boykov04. The core of this library is
the C++ implementation by Vladimir Kolmogorov, which
can be downloaded from his `homepage <http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/>`_.
Besides the wrapper to the C++ library, PyMaxflow offers

* NumPy integration, 
* methods for fast construction of common graph
  layouts in computer vision and graphics,
* implementation of algorithms for fast energy
  minimization which use the `maxflow` method: the αβ-swap
  and the α-expansion.

Take a look at the `PyMaxflow documentation <http://pmneila.github.com/PyMaxflow/>`_.

Requirements
------------

You need the following libraries installed on you system in order to
build PyMaxflow:

* `Boost.Python <http://www.boost.org/>`_
* `NumPy <http://numpy.scipy.org/>`_


Installation
------------

Edit the file ``environment.py`` according to the configuration
of your system. Change the ``ready`` variable to ``True`` when
finished. Then, open a terminal and write::

  $ python setup.py build
  ... lots of text ...

If everything went OK, you should be able to install the
package with::

  $ sudo python setup.py install


Alternative build
-----------------

If you do not have ``setuptools`` installed on your system,
an alternative build which uses CMake is available::

  $ cd maxflow/src
  $ mkdir build
  $ cd build
  $ cmake ../
  ... text ...

Edit the ``CMakeCache.txt`` if you want. It is important to
set the ``CMAKE_BUILD_TYPE`` variable to ``Release``. Then,
compile the code with::

  $ make
  ... text ...

The directory ``maxflow`` contains the full package now. You should
move it to a proper place included in your Python path, or change
the Python path accordingly.

Documentation
-------------

The documentation of the package is available under the ``doc``
directory. To generate the HTML documentation, use::

  $ cd doc/
  $ make html

