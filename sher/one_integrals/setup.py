from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#integral_utils
#mp2
#help_functions



setup(
    ext_modules=cythonize("integral_utils.pyx" , language_level = '3'),
    include_dirs=[numpy.get_include()]
)   