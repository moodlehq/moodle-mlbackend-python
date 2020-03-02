#!/usr/bin/env python3

# This file allows Moodle ML Backend to plug into a web server using
# Python's Web Server Gateway Interface (WSGI). There are many such
# servers to choose from.
#
# WSGI will scale much better than running webapp.py directly.

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from webapp import app as application

# uncomment the next line to get more information when debugging, but
# DON'T DO THIS IN PRODUCTION!
#application.debug = True
