#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from pathlib import Path

from setuptools import setup, find_packages

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / 'VERSION').read_text()

setup(
    name='mlrl-common',
    version=VERSION,
    description='Provides common modules to be used by different types of multi-label rule learning algorithms',
    long_description=(Path(__file__).resolve().parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Michael Rapp',
    author_email='michael.rapp.ml@gmail.com',
    url='https://github.com/mrapp-ke/Boomer',
    download_url='https://github.com/mrapp-ke/Boomer/releases',
    project_urls={
        'Documentation': 'https://mlrl-boomer.readthedocs.io/en/latest',
        'Issue Tracker': 'https://github.com/mrapp-ke/Boomer/issues'
    },
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords=[
        'machine learning',
        'scikit-learn',
        'multi-label classification',
        'rule learning'
    ],
    platforms=[
        'Linux'
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0'
    ],
    packages=find_packages(),
    package_data={
        "": ['*.so*']
    },
    zip_safe=True
)
