"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import os
import shutil

from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / 'VERSION').read_text()


class PrecompiledExtension(Extension):

    def __init__(self, name, path):
        super().__init__(name, [])
        self.name = name
        self.path = path


class PrecompiledExtensionBuilder(build_ext):

    def build_extension(self, ext):
        if isinstance(ext, PrecompiledExtension):
            build_dir = Path(self.get_ext_fullpath(ext.name)).parent
            target_file = Path(os.path.join(build_dir, ext.path))
            os.makedirs(target_file.parent, exist_ok=True)
            shutil.copy(ext.path, target_file)
        else:
            super().build_extension(ext)


def find_extensions(directory):
    extensions = []

    for path, _, file_names in os.walk(directory):
        for file_name in file_names:
            if '.so' in file_name or '.pyd' in file_name or '.dylib' in file_name or '.lib' in file_name \
                    or '.dll' in file_name:
                extension_path = Path(os.path.join(path, file_name))
                extension_name = file_name[:file_name.find('.')]
                extensions.append(PrecompiledExtension(extension_name, extension_path))

    return extensions


setup(name='mlrl-boomer',
      version=VERSION,
      description='A scikit-learn implementation of BOOMER - an algorithm for learning gradient boosted multi-label '
      'classification rules',
      long_description=(Path(__file__).resolve().parent.parent.parent.parent / 'README.md').read_text(),
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
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 3',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords=[
          'machine learning',
          'scikit-learn',
          'multi-label classification',
          'rule learning',
          'gradient boosting',
      ],
      platforms=[
          'Linux',
          'MacOS',
          'Windows',
      ],
      python_requires='>=3.8',
      install_requires=[
          'mlrl-common==' + VERSION,
      ],
      packages=find_packages(),
      ext_modules=find_extensions('mlrl'),
      cmdclass={'build_ext': PrecompiledExtensionBuilder},
      zip_safe=True)
