# -*- encoding: utf-8 -*-

from distutils.core import setup
import runpy
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

# Get the version number.
__version_str__ = runpy.run_path("maxflow/version.py")["__version_str__"]

numpy_include_dir = numpy.get_include()

maxflow_module = Extension(
    "maxflow._maxflow",
    [
        "maxflow/src/_maxflow.pyx",
        "maxflow/src/core/maxflow.cpp",
        "maxflow/src/fastmin.cpp"
    ],
    language="c++",
    include_dirs=[
        numpy_include_dir,
    ]
)

setup(
    name="PyMaxflow",
    version=__version_str__,
    description="A mincut/maxflow package for Python",
    author="Pablo Márquez Neila",
    author_email="pablo.marquezneila@epfl.ch",
    url="https://github.com/pmneila/PyMaxflow",
    license="GPL",
    long_description="""
    PyMaxflow is a Python library for graph construction and
    maxflow computation (commonly known as `graph cuts`). The
    core of this library is the C++ implementation by
    Vladimir Kolmogorov, which can be downloaded from his
    `homepage <http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/>`_.
    Besides the wrapper to the C++ library, PyMaxflow offers

    * NumPy integration,
    * methods for the construction of common graph
      layouts in computer vision and graphics,
    * implementation of algorithms for fast energy
      minimization which use the `maxflow` method:
      the alpha-beta-swap and the alpha-expansion.

    """,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    packages=["maxflow"],
    ext_modules=cythonize([maxflow_module]),
    requires=['numpy', 'Cython']
)
