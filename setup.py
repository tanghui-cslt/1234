# This file is part of THIS AWESOME  PAPER [1].
# Copyright (C) 2018-THESE AWESOME AUTHORS
#
# GENEO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (mattiagbergomi@gmail.com) or
# use the tools available at https://gitlab.com/mattia.bergomi.
#
# [1]

import os
import sys
from distutils.sysconfig import get_python_lib
from setuptools import find_packages, setup
import subprocess

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 5)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of geneo requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

requirements = ['tensorflow>=1.9.0',
                'keras>=2.2.0',
                'matplotlib>=2.2.2',
                'seaborn>=0.9.0',
                'opencv-python>=3.4.2.17',
                'tqdm>=4.23.4',
                'imutils>=0.4.6',
                'dionysus>=2.0.6',
                'parmap>=1.5.1',
                'Sphinx>=1.7.6',
                'sphinx-autobuild>=0.7.1',
                'sphinx-rtd-theme>=0.4.1',
                'scikit-learn>=0.20.0']
EXCLUDE_FROM_PACKAGES = []

setup(
    name='geneo',
    version='0.0.0-prealpha',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='',
    author='',
    author_email='',
    description=(''),
    license='GNU General Public License v3 or later (GPLv3+)',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    entry_points={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Machine Learning',
        'Topic :: Scientific/Engineering :: Data Analysis',
        'Topic :: Scientific/Engineering :: Topology',
    ],
)
