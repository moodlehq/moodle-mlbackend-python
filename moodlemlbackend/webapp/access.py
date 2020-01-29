import os
import re

from functools import wraps

from flask import request


STASH_DIR = os.environ.get('MOODLE_ML_STASH_DIR')


def stash_data(*args, **kwargs):
    import pickle, time, random
    reqdir = os.path.join(STASH_DIR, 'requests')
    os.makedirs(reqdir, exist_ok=True)
    t = time.strftime('%Y-%m-%d-%H-%M-%S-')
    s = os.path.join(reqdir, t + re.sub(r'\W', '_', request.path))
    try:
        f = open(s, 'xb')
    except FileExistsError:
        for i in range(1, 10):
            s2 = f'{s}.{i}'
            try:
                f = open(s2, 'xb')
                break
            except FileExistsError:
                continue
    r = {
        'url':   request.url,
        'data':  request.get_data(),
        'headers': request.headers.to_wsgi_list()
    }
    pickle.dump(r, f)
    f.close()


def check_access(f):
    '''Checks the access to the route.'''

    @wraps(f)
    def access_wrapper(*args, **kwargs):
        if STASH_DIR is not None:
            stash_data(*args, **kwargs)

        # Check that the environment var is properly set.
        envvarname = "MOODLE_MLBACKEND_PYTHON_USERS"
        if envvarname not in os.environ:
            raise Exception(
                envvarname + ' environment var is not set in the server.')

        if re.search(os.environ[envvarname], r'[^A-Za-z0-9_\-,:$]'):
            raise Exception(
                'The value of ' + envvarname + ' environment should be '
                'a list of colon separated user/password values.\n'
                'Usernames and passwords can contain letters, numbers, '
                'and the symbols "$_-".\n'
                'Like this:\n'
                '  "user1:password1,user2:password2,user_3:pa$$word3"')

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
