import os
import re

from functools import wraps

from flask import request


def check_access(f):
    '''Checks the access to the route.'''

    @wraps(f)
    def access_wrapper(*args, **kwargs):

        # Check that the environment var is properly set.
        envvarname = "MOODLE_MLBACKEND_PYTHON_USERS"
        if envvarname not in os.environ:
            raise Exception(
                envvarname + ' environment var is not set in the server.')

        if re.match(os.environ[envvarname], '[^A-Za-z0-9_\-,$]'):
            raise Exception(
                'The value of ' + envvarname + ' environment var does not ' +
                ' adhere to [^A-Za-z0-9_\-,$]')

        users = os.environ[envvarname].split(',')

        if (request.authorization is None or
                request.authorization.username is None or
                request.authorization.password is None):
            # Response for the client.
            return 'No user and/or password included in the request.', 401

        for user in users:
            userdata = user.split(':')
            if len(userdata) != 2:
                raise Exception('Incorrect format for ' +
                                envvarname + ' environment var. It should ' +
                                'contain a comma-separated list of ' +
                                'username:password.')

            if (userdata[0] == request.authorization.username and
                    userdata[1] == request.authorization.password):

                # If all good we return the return from 'f' passing the
                # original list of params to it.
                return f(*args, **kwargs)

        # Response for the client.
        return 'Incorrect user and/or password provided by Moodle.', 401

    return access_wrapper
