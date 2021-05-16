# Copyright (C) 2021 poypoyan

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="edhsmm",
    version="0.1.2",
    description="An(other) implementation of Explicit Duration HMM/HSMM in Python 3",
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
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["edhsmm"],
    ext_modules=cythonize("edhsmm/hsmm_core_x.pyx",
                include_path = [numpy.get_include()]),
    python_requires=">=3.5",   # compatibility with hmmlearn
    install_requires=[
        "numpy>=1.10",   # compatibility with hmmlearn
        "scikit-learn>=0.16",   # sklearn.utils.check_array
        "scipy>=0.19",   # scipy.special.logsumexp
    ],
)