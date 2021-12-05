.. _compilation:

Building from Source
--------------------

As discussed in the previous section :ref:`structure`, the algorithm that is provided by this project is mostly implemented in `C++ <https://en.wikipedia.org/wiki/C%2B%2B>`__ to ensure maximum efficiency. In addition, a `Python <https://en.wikipedia.org/wiki/Python_(programming_language)>`__ wrapper that integrates the algorithm with the `scikit-learn <https://scikit-learn.org>`__ framework is provided. To make the underlying C++ implementation accessible from within the Python code, `Cython <https://en.wikipedia.org/wiki/Cython>`__ is used.

Unlike pure Python programs, the C++ and Cython source files must be compiled for a particular target platform. To ease the process of compiling the source code, the project comes with a Makefile that automates the necessary steps. In the following, we discuss the individual steps that are necessary for building the project from scratch.

.. note::
    The Makefile that is provided for the compilation of the software package is intended for use on Linux systems. Although compilation should be possible on Windows and MacOS systems, we currently do not officially support compilation on these platforms.

As a prerequisite, Python 3.7 (or a more recent version) must be available on the host system. All remaining compile- or build-time dependencies will automatically be installed when following the instructions below.

**Step 1: Creating a virtual environment**

The build process is based on creating a virtual Python environment that allows to install build-time dependencies in an isolated manner and independently from the host system. Once all packages have successfully been built, they are installed into the virtual environment. To create new virtual environment and install all necessarily build-time dependencies, the following command must be executed:

.. code-block:: text

   make venv

All compile-time dependencies (`numpy`, `scipy`, `cython`, `meson`, `ninja`, etc.) that are required for building the project should automatically be installed into the virtual environment when executing the above command. As a result, a subdirectory `venv/` should have been created in the project's root directory.

**Step 2: Compiling the C++ code**

Once a new virtual environment has successfully been created, the compilation of the C++ code can be started by executing the following command:

.. code-block:: text

   make compile_cpp

Compilation is based on the build system `Meson <https://mesonbuild.com/>`_ and uses `Ninja <https://ninja-build.org/>`_ as a backend. After the above command has been completed, a new directory `cpp/build/` should have been created. It contains the shared libraries ("libmlrlcommon", "libmlrlboosting" and possibly others) that provide the basic functionality of the project's algorithms.

**Step 3: Compiling the Cython code**

Once the compilation of the C++ code has completed, the Cython code that allows to access the corresponding shared libraries from within Python can be compiled in the next step. Again, Meson and Ninja are used for compilation. It can be started via the following command:

.. code-block:: text

   make compile_cython

As a result of executing the above command, the directory `python/build` should have been created. It contains Python extension modules for the respective target platform.

.. note::
    Instead of performing the previous steps one after the other, the command ``make compile`` can be used to compile the C++ and Cython source files in a single step.

**Step 4: Copying compilation files into the Python source tree**

The shared library files and Python extension modules that have been created in the previous steps must afterwards be copied into the source tree that contains the Python code. This can be achieved by executing the following commands:

.. code-block:: text

   make install_cpp
   make install_cython

This should result in the compilation files, which were previously located in the `cpp/build/` and `python/build/` directories, to be copied into the `cython/` subdirectories that are contained by each Python module (e.g., into the directory `python/subprojects/common/mlrl/common/cython/`).

**Step 5: Building wheel packages**

Once the compilation files have been copied into the Python source tree, wheel packages can be built for the individual Python modules via the following command:

.. code-block:: text

   make wheel

This should result in .whl files being created in a new `dist/` subdirectory inside the directories that correspond to the individual Python modules (e.g., in the directory `python/subprojects/common/dist/`).

**Step 6: Installing the wheel packages into the virtual environment**

The wheel packages that have previously been created, as well as its runtime-dependencies (e.g., `scikit-learn` or `liac-arff`), can finally be installed into the virtual environment via the following command:

.. code-block:: text

   make install

After this final step has completed, the Python packages can be used from within the virtual environment. To ensure that the installation of the wheel packages was successful, check if a `mlrl/` directory has been created in the `lib/` directory of the virtual environment (depending on the Python version, it should be located at `venv/lib/python3.9/site-packages/mlrl/` or similar). If this is the case, the algorithm can be used from within your own Python code. Alternatively, the command line API can be used to start an experiment (see :ref:`experiments`).

.. warning::
    Whenever any C++, Cython or Python source files have been modified, they must be recompiled and updated wheel packages must be installed into the virtual environment by executing the command ``make install``. If any compilation files do already exist, this will only result in the affected parts of the code to be rebuilt.

**Cleanup**

To get rid of any compilation files, as well as of the generated wheel packages, the following command can be used:

.. code-block:: text

   make clean_install

If you want to delete the virtual environment as well, you should use the following command instead:

.. code-block:: text

   make clean

.. note::
    For even more fine-grained control, the Makefile allows to delete the files that result from the individual steps that have been described above. This includes the command ``clean_wheel`` for deleting the wheel packages that are located in the individual `dist/` directories, the commands ``make clean_cython_install`` and ``make clean_cpp_install`` for removing the Cython or C++ compilation files from the Python source tree, as well as ``make clean_cython`` and ``make clean_cpp`` for removing the Cython or C++ compilation files from the respective `build/` directories.
