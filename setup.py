# -*- encoding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import runpy
from distutils.extension import Extension

# Get the version number.
__version_str__ = runpy.run_path("maxflow/version.py")["__version_str__"]

# Lazy evaluate extension definition, to allow correct requirements install
class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())


def extensions():
    import numpy
    from Cython.Build import cythonize
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
    return cythonize([maxflow_module])


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
    ext_modules=lazy_cythonize(extensions),
    requires=['numpy', 'Cython'],
    setup_requires=['numpy', 'Cython']
)
