.. _compilation:

Building from Source
--------------------

As discussed in the previous section :ref:`structure`, the algorithm that is provided by this project is mostly implemented in `C++ <https://en.wikipedia.org/wiki/C%2B%2B>`__ to ensure maximum efficiency (requires C++ 14 or newer). In addition, a `Python <https://en.wikipedia.org/wiki/Python_(programming_language)>`__ wrapper that integrates the algorithm with the `scikit-learn <https://scikit-learn.org>`__ framework is provided (requires Python 3.8 or newer). To make the underlying C++ implementation accessible from within the Python code, `Cython <https://en.wikipedia.org/wiki/Cython>`__ is used (requires Cython 29 or newer).

Unlike pure Python programs, the C++ and Cython source files must be compiled for a particular target platform. To ease the process of compiling the source code, the project comes with a `Makefile <https://en.wikipedia.org/wiki/Make_(software)>`__ that automates the necessary steps. In the following, we discuss the individual steps that are necessary for building the project from scratch. This is necessary if you intend to modify the library's source code. If you want to use the algorithm without any custom modifications, the :ref:`installation` of pre-built packages is usually a better choice.

**Prerequisites**

As a prerequisite, a supported version of Python, a suitable C++ compiler, as well as an implementation of the Make build automation tool, must be installed on the host system. The installation of these software components depends on the operation system at hand. In the following, we provide installation instructions for the supported platforms.

* **Linux:** Nowadays, most Linux distributions include a pre-installed version of Python 3. If this is not the case, instructions on how to install a recent Python version can be found in Python's `Beginners Guide <https://wiki.python.org/moin/BeginnersGuide/Download>`__. As noted in this guide, Python should be installed via the distribution's package manager if possible. The most common Linux distributions do also ship with `GNU Make <https://www.gnu.org/software/make/>`__ and the `GNU Compiler Collection <https://gcc.gnu.org/>`__ (GCC) by default. If this is not the case, these software packages can typically be installed via the distribution's default package manager.
* **MacOS:** Recent versions of MacOS do not include Python by default. A suitable Python version can manually be downloaded from the `project's website <https://www.python.org/downloads/macos/>`__. Alternatively, the package manager `Homebrew <https://en.wikipedia.org/wiki/Homebrew_(package_manager)>`__ can be used for installation via the command ``brew install python``. MacOS relies on the `Clang <https://en.wikipedia.org/wiki/Clang>`__ compiler for building C++ code by default. It is part of the `Xcode <https://developer.apple.com/support/xcode/>`__ developer toolset. In addition, for proper multi-threading support, the `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`__ library must be installed. We recommend to install it via Homebrew by running the command ``brew install libomp``.
* **Windows:** Python releases for Windows are available at the `project's website <https://www.python.org/downloads/windows/>`__. In addition, an implementation of the Make tool must be installed. We recommend to use `GNU Make for Windows <http://gnuwin32.sourceforge.net/>`__. For the compilation of the project's source code, the MSVC compiler must be used. It is included in the `Build Tools for Visual Studio <https://visualstudio.microsoft.com/downloads/>`__. Finally, `Powershell <https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/powershell>`__ must be used to run the project's Makefile. It should be included by default on modern Windows systems.

Additional compile- or build-time dependencies will automatically be installed when following the instructions below and must not be installed manually.

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

The Makefile allows to delete the files that result from the individual steps that have been described above. To delete the wheel packages that have been created via the command ``make wheel`` the following command can be used:

.. code-block:: text

   make clean_wheel

The following command allows to remove the shared library files and Python extension modules that have been copied into the Python source tree via the commands ``make install_cpp`` and ``make install_cython``:

.. code-block:: text

   make clean_install

The commands ``make clean_cython`` and ``make clean_cpp`` remove the Cython or C++ compilation files that have been created via the command ``make compile_cython`` or ``make compile_cpp`` from the respective `build/` directories. If you want to delete both, the Cython and C++ compilation files, the following command can be used:

.. code-block:: text

   make clean_compile

.. note::
    If you want to delete all compilation files that have been created via the Makefile, including the virtual environment, you should use the command ``make clean``.
