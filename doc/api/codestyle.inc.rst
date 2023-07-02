.. _codestyle:

Code Style
----------

We aim to enforce a consistent code style across the entire project. For formatting the C++ code, we employ `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`__. The desired C++ code style is defined in the file ``.clang-format`` in project's root directory. Accordingly, we use `YAPF <https://github.com/google/yapf>`__ to enforce the Python code style defined in the file ``.style.yapf``. In addition, to keep the ordering of imports in Python and Cython source files consistent, `isort <https://github.com/PyCQA/isort>`__ is applied using the configuration in the file ``.isort.cfg``. If you have modified the project's source code, you can check whether it adheres to our style guidelines via the following command:

.. code-block:: text

   make test_format

.. note::
    If you want to check for compliance with the C++ or Python code style independently, you can alternatively use the command ``make test_format_cpp`` or ``make test_format_python``.

In order to automatically format the project's source files according to our style guidelines, the following command can be used:

.. code-block:: text

   make format

.. note::
    If you want to format only the C++ source files, you can run the command ``make format_cpp`` instead. Accordingly, the command ``make format_python`` may be used to format only the Python source files.

Whenever any source files have been modified, a `Github Action <https://docs.github.com/en/actions>`__ is run automatically to verify if they adhere to our code style guidelines. The result of these runs can be found in the `Github repository <https://github.com/mrapp-ke/Boomer/actions>`__.
