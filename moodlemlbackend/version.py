"""Utility module to print the package version"""

import os


def print_version():
    """Prints moodlemlbackend package version"""

    # Version read from file.
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = open(os.path.join(here, 'VERSION'))
    version = version_file.read().strip()

    print(version)

print_version()
