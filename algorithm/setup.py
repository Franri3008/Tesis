from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "localsearches",
        ["localsearches.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "initial_solutions",
        ["initial_solutions.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "evaluation",
        ["evaluation.pyx"],
        include_dirs=[numpy.get_include()]
    )
];

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"})
);