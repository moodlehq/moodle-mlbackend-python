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
version_file = open(os.path.join(here, 'moodlemlbackend', 'VERSION'))
version = version_file.read().strip()

setup(
    name='moodlemlbackend',

    version=version,

    description='Python machine learning backend used by mlbackend_python Moodle plugin',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/moodlehq/moodle-mlbackend-python',

    # Author details
    author='Moodle HQ',
    author_email='integration@moodle.com',

    # Choose your license
    license='GPLv3',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Topic :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='moodle machine learning numpy scikit-learn tensorflow',

    packages=find_packages(),
    package_data={
        'moodlemlbackend': ['VERSION']
    },
    install_requires=[
        'matplotlib>=3.0,<3.1',
        'numpy>=1.16.0,<1.17',
        'scikit-learn>=0.21,<0.22',
        'joblib>=0.13.0,<0.14',
        'tensorflow>=1.14.0,<1.15',
        'flask>=1.0.2,<1.1',
    ],
)
