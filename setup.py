# -*- encoding: utf-8 -*-

from setuptools import setup

import runpy
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

# Get the version number.
__version_str__ = runpy.run_path("maxflow/version.py")["__version_str__"]


def extensions():
    numpy_include_dir = numpy.get_include()
    maxflow_module = Extension(
        "maxflow._maxflow",
        [
            "maxflow/src/_maxflow.pyx",
            "maxflow/src/core/maxflow.cpp",
            "maxflow/src/fastmin.cpp"
        ],
        language="c++",
        extra_compile_args=['-std=c++11', '-Wall'],
        include_dirs=[numpy_include_dir],
        depends=[
            "maxflow/src/fastmin.h",
            "maxflow/src/grid.h",
            "maxflow/src/pyarray_symbol.h",
            "maxflow/src/pyarraymodule.h",
            "maxflow/src/core/block.h",
            "maxflow/src/core/graph.h"
        ],
        # Cython 0.29 generates code using the deprecated pre 1.7 NumPy API
        # This line should be uncommented when Cython 3.0 is officially released
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
    return cythonize(
        [maxflow_module],
        language_level="3"
    )


setup(
    name="PyMaxflow",
    version=__version_str__,
    description="A mincut/maxflow package for Python",
    author="Pablo Márquez Neila",
    author_email="pablo.marquez@unibe.ch",
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
    ext_modules=extensions(),
    install_requires=['numpy']
)
