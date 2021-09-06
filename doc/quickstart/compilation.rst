Building the Project
--------------------

The algorithm provided by this project is mostly implemented in C++. In addition, a Python wrapper that implements the scikit-learn API is provided. To integrate the underlying C++ implementation with Python, `Cython <https://cython.org>`_ is used.

Unlike pure Python programs, the code written in C++ and Cython must be compiled to be able to run the algorithm. To facilitate the compilation, the project comes with a Makefile that automatically executes the necessary steps.

As a prerequisite, Python 3.7 (or a more recent version) must be available on the host system. All remaining compile- or build-time dependencies will automatically installed when following the instructions below.

.. note::
    We only support x86_64 Linux platforms out-of-the-box, although compilation should be possible on Windows and MacOS systems as well. Unfortunately, we currently do not have the resources to provide support for these platforms. For future releases we plan to distribute prebuilt packages for all major platforms.

**Step 1: Create a virtual environment**

At first, a virtual Python environment can be created via the following command:

.. code-block:: text

   make venv

All compile-time dependencies (`numpy`, `scipy`, `Cython`, `meson` and `ninja`) that are required for building the project should automatically be installed into the virtual environment when executing the above command. As a result, a subdirectory "venv" should have been created in the project's root directory.

**Step 2: Compilation**

Afterwards, the compilation can be started by executing the following command:

.. code-block:: text

   make compile

Compilation is based on the build system `Meson <https://mesonbuild.com/>`_ and uses `Ninja <https://ninja-build.org/>`_ as a backend.

Whenever any C++ or Cython source files have been modified, they must be recompiled by running the above command again. If any compilation files do already exist, only the affected parts of the code will be recompiled.

**Step 3: Installation**

Once the compilation has completed, the library can be installed into the virtual environment. For this purpose, the project's Makefile provides the following command:

.. code-block:: text

   make install

The above command does also install all runtime dependencies, such as `scikit-learn`. A full list of all dependencies can be found in the file "python/setup.py". 

**Step 4: Generating the Documentation (Optional)**

In order to generate the documentation (this document), `Doxygen <https://sourceforge.net/projects/doxygen/>`_ must be installed on the system beforehand. It is used to automatically generate an API documentation from the source code. By running the following command, the documentation's HTML documents are generated:

.. code-block:: text

   make doc 

Afterwards, the generated files can be found in the directory `doc/build_/html`.

**Cleanup**

To get rid of any compilation files, the generated documentation files, as well as of the virtual environment, the following command can be used:

.. code-block:: text

   make clean
 

For more fine-grained control, the command ``make clean_venv`` can be used for deleting the virtual environment. The command `make clean_compile` does only delete the compilation files. If only the compiled Cython files should be removed, the command ``make clean_cython`` can be used. Accordingly, the command ``make clean_cpp`` removes the compiled C++ files. To delete the generated documentation files, the command ``make clean_doc`` may be used.

