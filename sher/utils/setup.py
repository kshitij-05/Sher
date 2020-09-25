from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#integrals_functions
#mp2
#help_functions



setup(
    ext_modules=cythonize("enuc.pyx" , language_level = '3'),
    include_dirs=[numpy.get_include()]
)   
