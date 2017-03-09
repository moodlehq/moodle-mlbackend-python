# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Version read from file.
version_file = open(os.path.join(here, 'VERSION'))
version = version_file.read().strip()

setup(
    name='moodleinspire',

    version=version,

    description='Python predictions processor backend for Moodle Inspire',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/dmonllao/moodleinspire-python-backend',

    # Author details
    author='David Monllao',
    author_email='davidm@moodle.com',

    # Choose your license
    license='GPLv3',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Topic :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='moodle machine learning numpy scikit-learn tensorflow',

    packages=find_packages(),
    install_requires=[
        'matplotlib>=1.5.0,<1.6',
        'numpy>=1.11.0,<1.12',
        'scikit-learn>=0.17.0,<0.18',
        'tensorflow>=1.0.0<1.1',
    ],
)
