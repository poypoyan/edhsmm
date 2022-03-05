# Copyright (C) 2022 poypoyan

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [Extension("edhsmm._hsmm_core", ["edhsmm/_hsmm_core.pyx"],
              include_dirs = [numpy.get_include()])]


setup(
    name="edhsmm",
    version="0.2.1",
    description="An(other) implementation of Explicit Duration Hidden Semi-Markov Models in Python 3",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="poypoyan",
    author_email="srazin123@gmail.com",
    url="https://github.com/poypoyan/edhsmm",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
    ],
    packages=["edhsmm"],
    ext_modules=cythonize(extensions),
    python_requires=">=3.5",   # compatibility with hmmlearn
    install_requires=[
        "numpy>=1.17",   # np.random.default_rng
        "scikit-learn>=0.16",   # sklearn.utils.check_array
        "scipy>=0.19",   # scipy.special.logsumexp
    ],
)