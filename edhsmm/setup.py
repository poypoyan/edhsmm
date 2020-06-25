# Only compiles hsmm_core_x.pyx
# Use this command: python setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="hsmm_core_x",
    include_dirs = [numpy.get_include()],
    ext_modules=cythonize("hsmm_core_x.pyx"),
    zip_safe=False,
)