cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(molssi_sss_pybind)

# This must be run using a conda environment
if("$ENV{CONDA_PREFIX}" STREQUAL "")
    message(FATAL_ERROR "This must be run within the MolSSI SSS conda prefix. "
                        "Delete the build directory and try again")
endif()

# CMake will sometimes find the conda python version
# (rather the the python inside the sss environment)
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3)

# CXX flags
set (CMAKE_CXX_FLAGS "-Ofast -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -fopenmp -I${CONDA_PREFIX}/include")

# Find the pybind11 in the conda path
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 CONFIG REQUIRED
             PATHS $ENV{CONDA_PREFIX}
             NO_DEFAULT_PATH)

message(STATUS "Found pybind11: ${pybind11_CONFIG}")

# Creates a python module named "module_name"
# pybind11_add_module(module_name MODULE file1.cpp file2.cpp)
pybind11_add_module(xform MODULE xform.cpp)
