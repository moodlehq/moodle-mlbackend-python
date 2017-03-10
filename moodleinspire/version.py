import os
import sys

# Version read from file.
here = os.path.abspath(os.path.dirname(__file__))
version_file = open(os.path.join(here, 'VERSION'))
version = version_file.read().strip()

print(version)
sys.exit(0)
