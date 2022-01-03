Contribute
==========

If you consider contributing to biapol-utilities (thanks in advance!), please follow these guidelines.

Docstring formatting
-----------------------

In general, biapol-utilites abides to the numpy docstring standards, as described `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_. This style is implemented as a docstring template in most Python IDEs (e.g., Spyder, PyCharm, etc.).

Re-using code
-----------------------

Biapol-utilities is intented to be a wrapper/collection library for useful image-processing utility functions, that may span several repositories. If you re-use code from third-party libraries (e.g., scikit-image or scikit-learn), please copy the license of the respective (`directory in the repository <https://github.com/BiAPoL/biapol-utilities/tree/create-sphinx-doc/license_thirdparty>`_). Feel free to also add sources as footnotes to the docstrings as described `here <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#footnote>`_ .


Build the documentation
-----------------------

Prior to submitting pull requests to biapol-utils, make sure to (re-)build the documentation. To do so, you need to install additional packages:

.. prompt:: bash $

    pip install sphinx nbsphinx sphinx-prompt
    conda install pandoc

To build the documentation, you need to be in the ``docs/`` folder:

.. prompt:: bash $

    cd docs/

Generate the html files for the page by running

.. prompt:: bash $

    make html

The documentation will then be generated in the ``_build/html`` directory. This will run all the examples, which may take a while. Upon submitting the pull request, the page will then automatically be built and hosted on github pages once the PR is accepted.

*Note*: Please make sure that the documentation html files are built properly before submitting a pull request. Once your changes are pushed to the main branch, the documentation homepage will automatially be rebuilt from the changed source code.
