"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / 'VERSION').read_text()


def find_dependencies(requirements_file, dependency_names):
    requirements = {
        requirement.key: requirement
        for requirement in parse_requirements(requirements_file.read_text().split('\n'))
    }
    dependencies = []

    for dependency_name in dependency_names:
        match = requirements.get(dependency_name)

        if match is None:
            raise RuntimeError('Failed to determine required version of dependency "' + dependency_name + '"')

        dependencies.append(str(match))

    return dependencies


setup(name='mlrl-testbed',
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
          'Issue Tracker': 'https://github.com/mrapp-ke/Boomer/issues',
      },
      license='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords=[
          'machine learning',
          'scikit-learn',
          'multi-label classification',
          'rule learning',
          'evaluation',
      ],
      platforms=['any'],
      python_requires='>=3.7',
      install_requires=[
          'mlrl-common==' + VERSION,
          *find_dependencies(requirements_file=Path(__file__).resolve().parent.parent.parent / 'requirements.txt',
                             dependency_names=['liac-arff', 'tabulate']),
      ],
      extras_require={
          'BOOMER': ['mlrl-boomer==' + VERSION],
      },
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'boomer=mlrl.testbed.main_boomer:main [BOOMER]',
          ]
      },
      zip_safe=True)
