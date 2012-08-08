#! /usr/bin/env python

descr   = """Bootstrap Scikit

Algorithms for SciPy to calculate bootstrap confidence intervals for statistics functions
applied to data.
"""

import os
import sys

DISTNAME            = 'scikits.bootstrap'
DESCRIPTION         = 'Bootstrap confidence interval estimation routines for SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Constantine Evans',
MAINTAINER_EMAIL    = 'cevans@evanslabs.org',
URL                 = 'http://github.org/cgevans/scikits-bootstrap'
LICENSE             = 'Modified BSD'
DOWNLOAD_URL        = URL
VERSION             = '0.2dev'

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name, parent_package, top_path,
                           version = VERSION,
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    return config

if __name__ == "__main__":
    setup(configuration = configuration,
        install_requires = ['numpy','scipy'],
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        zip_safe = True,
        classifiers =
            [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              ])
