.. _testing:

Testing the Code
----------------

To be able to detect problems with the project's source code early during development, it comes with a large number of integration tests. Each of these tests runs a different configuration of the project's algorithms via the command line API and checks for unexpected results. If you want to execute the integrations tests on your own system, you can use the following command:

.. code-block:: text

   make tests

The integration tests are also run automatically on a `CI server <https://en.wikipedia.org/wiki/Continuous_integration>`__ whenever relevant parts of the source code have been modified. For this purpose, we rely on the infrastructure provided by `Github Actions <https://docs.github.com/en/actions>`__. A track record of past test runs can be found in the `Github repository <https://github.com/mrapp-ke/Boomer/actions>`__.
