import os
import re

from functools import wraps

from flask import request


class MoodleMLError(Exception):
    pass


STASH_DIR = os.environ.get('MOODLE_ML_STASH_DIR')

USER_ENV = "MOODLE_MLBACKEND_PYTHON_USERS"
USERS = {}

def _init_users():
    users = os.environ.get(USER_ENV)
    if users is None:
        raise MoodleMLError(f'The value of {USER_ENV} environment should be '
                            'a comma separated list of colon separated '
                            'user/password values.\n'
                            'Usernames and passwords can contain letters, '
                            'numbers, and the symbols "$_-".\n'
                            'Like this:\n'
                            '  "user1:passwd1,user2:passwd2,user_3:pa$$word3"')

    for userpass in users.split(','):
        user, password = userpass.split(':', 1)

        # why this assertion? well it matches the existing behaviour, and you
        # won't believe what happens in the next commit!
        assert ':' not in password

        USERS[user] = password


_init_users()


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

        if (request.authorization is None or
                request.authorization.username is None or
                request.authorization.password is None):
            # Response for the client.
            return 'No user and/or password included in the request.', 401

        passwd = USERS.get(request.authorization.username)
        if passwd == request.authorization.password:
                # If all good we return the return from 'f' passing the
                # original list of params to it.
                return f(*args, **kwargs)

        # Response for the client.
        return 'Incorrect user and/or password provided by Moodle.', 401

    return access_wrapper
