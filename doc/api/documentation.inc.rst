.. _documentation:

Generating the Documentation
----------------------------

In order to generate the documentation (this document), `Doxygen <https://sourceforge.net/projects/doxygen/>`_ must be installed on the host system beforehand. It is used to generate an API documentation from the C++ source files. By running the following command, the C++ API documentation is generated via Doxygen, the Python API documentation is created via `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`_ and the documentation's HTML files are generated via `sphinx <https://www.sphinx-doc.org/en/master/>`_:

.. code-block:: text

   make doc

Afterwards, the generated files can be found in the directory `doc/build_/html/`.

To clean up the generated documentation files, the following command can be used:

.. code-block:: text

   make clean_doc
