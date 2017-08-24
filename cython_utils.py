__author__ = 'ando'
from distutils.core import setup
from Cython.Build import cythonize
import numpy

# file used to compile the Cython code
# use the command 'python cython_utils build_ext --inplace' to build the needed code


setup(
    name = 'SDG app',
    ext_modules=cythonize("utils/training_sdg_inner.pyx", compiler_directives={'boundscheck': False,'wraparound': False, 'cdivision': True}),
    include_dirs=[numpy.get_include()]
)