
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Maxflow)

SET(Boost_USE_STATIC_LIBS   OFF)
SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost 1.42.0 REQUIRED COMPONENTS python)
FIND_PACKAGE(PythonLibs REQUIRED)
#FIND_PACKAGE(OpenMP REQUIRED)

# Set the numpy headers dir.
EXECUTE_PROCESS(COMMAND python -c "import numpy; print numpy.get_include()"
                RESULT_VARIABLE NUMPY_RESULT
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
                ERROR_VARIABLE NUMPY_ERROR)
IF(${NUMPY_RESULT} EQUAL 1)
    MESSAGE(SEND_ERROR ${NUMPY_ERROR})
ENDIF(${NUMPY_RESULT} EQUAL 1)

SET(maxflow_SOURCES core/block.h core/graph.h core/graph.cpp core/maxflow.cpp
		esmooth.h grid.h fastmin.h
        python.cpp abswap.cpp aexpansion.cpp pyarray_index.cpp)

#SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
ADD_LIBRARY(_maxflow MODULE ${maxflow_SOURCES})
SET_TARGET_PROPERTIES(_maxflow PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_maxflow ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
