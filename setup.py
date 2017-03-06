# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='moodleinspire',

    version='0.0.1',

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
        'matplotlib==1.5.1',
        'numpy==1.11',
        'scikit-learn==0.17.1',
        'tensorflow==1.0.0rc2',
    ],
)
