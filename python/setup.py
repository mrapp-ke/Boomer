import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = True

# True, if all Cython compiler optimizations should be disabled
DEBUG = False

extensions = [
    Extension(name='*', sources=['**/*.pyx'], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
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
      python_requires='>=3.7',
      ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE, compiler_directives=compiler_directives),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
