#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from pathlib import Path

from setuptools import setup, find_packages

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / 'VERSION').read_text()

setup(
    name='mlrl-testbed',
    version=VERSION,
    description='Provides utilities for the training and evaluation of multi-label rule learning algorithms',
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
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords=[
        'machine learning',
        'scikit-learn',
        'multi-label classification',
        'rule learning',
        'evaluation'
    ],
    platforms=['any'],
    python_requires='>=3.7',
    install_requires=[
        'liac-arff>=2.5.0',
        'mlrl-common==' + VERSION
    ],
    extras_require={
        'BOOMER': ['mlrl-boomer==' + VERSION],
        'SECO': ['mlrl-seco==' + VERSION]
    },
    packages=find_packages(),
    entry_points='''
        [console_scripts]
        boomer = mlrl.testbed.main_boomer:main [BOOMER]
        seco = mlrl.testbed.main_seco:main [SECO]
    ''',
    zip_safe=True
)
