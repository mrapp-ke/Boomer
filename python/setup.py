import sys

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = True

# True, if all Cython compiler optimizations should be disabled
DEBUG = False

# The compiler/linker argument to enable OpenMP support
COMPILE_FLAG_OPEN_MP = '/openmp' if sys.platform.startswith('win') else '-fopenmp'

sources = [
    '**/*.pyx',
    'boomer/common/cpp/sparse.cpp',
    'boomer/common/cpp/predictions.cpp',
    'boomer/common/cpp/input_data.cpp',
    'boomer/common/cpp/statistics.cpp',
    'boomer/boosting/cpp/blas.cpp',
    'boomer/boosting/cpp/lapack.cpp',
    'boomer/boosting/cpp/label_wise_losses.cpp',
    'boomer/boosting/cpp/example_wise_losses.cpp',
    'boomer/boosting/cpp/statistics.cpp',
    'boomer/boosting/cpp/label_wise_statistics.cpp',
    'boomer/boosting/cpp/example_wise_statistics.cpp',
    'boomer/boosting/cpp/label_wise_rule_evaluation.cpp',
    'boomer/boosting/cpp/example_wise_rule_evaluation.cpp'
]

extensions = [
    Extension(name='*', sources=sources, language='c++', extra_compile_args=[COMPILE_FLAG_OPEN_MP],
              extra_link_args=[COMPILE_FLAG_OPEN_MP], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

compiler_directives = {
    'boundscheck': DEBUG,
    'wraparound': DEBUG,
    'cdivision': not DEBUG,
    'initializedcheck': DEBUG
}

setup(name='boomer',
      version='0.2.0',
      description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
      url='https://github.com/mrapp-ke/Boomer',
      author='Michael Rapp',
      author_email='mrapp@ke.tu-darmstadt.de',
      license='MIT',
      packages=['boomer'],
      install_requires=[
          "numpy>=1.19.0",
          "scipy>=1.5.0",
          "Cython>=0.29.0",
          'scikit-learn>=0.23.0',
          'scikit-multilearn>=0.2.0',
          'liac-arff>=2.4.0',
          'requests>=2.23.0'
      ],
      python_requires='>=3.8',
      ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE, compiler_directives=compiler_directives),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
