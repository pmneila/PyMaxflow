# -*- encoding: utf-8 -*-

import sys
from setuptools import setup, Extension
import numpy
import environment as env

# Get the version number.
ver_dict = {}
execfile("maxflow/version.py", ver_dict)
__version_str__ = ver_dict["__version_str__"]

if not env.ready and sys.argv[1] != 'sdist':
    print >> sys.stderr, "ERROR: The project is not configured. Please edit environment.py and follow the instructions."
    sys.exit(1)

numpy_include_dir = numpy.get_include()

maxflow_module = Extension('_maxflow',
                            include_dirs=[numpy_include_dir, env.boost_include_dir],
                            libraries=[env.boost_python_lib],
                            library_dirs=[env.boost_libs_dir],
                            sources=["maxflow/src/python.cpp", 
                                     "maxflow/src/abswap.cpp",
                                     "maxflow/src/aexpansion.cpp",
                                     "maxflow/src/pyarray_index.cpp",
                                     "maxflow/src/core/graph.cpp",
                                     "maxflow/src/core/maxflow.cpp"]
                          )

setup(name="PyMaxflow",
    version=__version_str__,
    description="A mincut/maxflow package for Python",
    author="Pablo Márquez Neila",
    author_email="p.mneila@upm.es",
    url="",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    packages=["maxflow"],
    ext_modules=[maxflow_module]
    )
