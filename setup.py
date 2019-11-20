#!/usr/bin/env python

from setuptools import setup, find_packages
setup(
    name = "scikits.bootstrap",
    version = "1.0.1",
    packages = find_packages(),

    install_requires = ['numpy', 'pyerf'],
    namespace_packages = ['scikits'],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.md', '*.rst'],
    },

    # metadata for upload to PyPI
    author = "Constantine Evans",
    author_email = "cevans@evanslabs.org",
    description = "Bootstrap confidence interval estimation routines for SciPy",
    license = "Modified BSD",
    #keywords = "",
    url = "http://github.com/cgevans/scikits-bootstrap",   # project home page, if any
    classifiers =
        [ 'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: Implementation :: PyPy',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering',
          'Operating System :: OS Independent',
          ],
    test_suite = "nose.collector"
    # could also include long_description, download_url, classifiers, etc.
)
